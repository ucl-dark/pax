import os
import time
from typing import Any, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import wandb
from pax.utils import MemoryState, TrainingState, load, save
from pax.watchers import (
    ipditm_stats,
    n_player_ipd_visitation,
    tensor_ipd_visitation,
)

MAX_WANDB_CALLS = 1000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class MFOSSample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray
    meta_actions: jnp.ndarray


@jax.jit
def reduce_outer_traj(traj: Sample) -> Sample:
    """Used to collapse lax.scan outputs dims"""
    # x: [outer_loop, inner_loop, num_opps, num_envs ...]
    # x: [timestep, batch_size, ...]
    num_envs = traj.observations.shape[2] * traj.observations.shape[3]
    num_timesteps = traj.observations.shape[0] * traj.observations.shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_timesteps, num_envs) + x.shape[4:]),
        traj,
    )


class MultishaperEvalRunner:
    """
    Reinforcement Learning runner provides a convenient example for quickly writing
    a MARL runner for PAX. The MARLRunner class can be used to
    run any two RL agents together either in a meta-game or regular game, it composes together agents,
    watchers, and the environment. Within the init, we declare vmaps and pmaps for training.
    Args:
        agents (Tuple[agents]):
            The set of agents that will run in the experiment. Note, ordering is
            important for logic used in the class.
        env (gymnax.envs.Environment):
            The environment that the agents will run in.
        save_dir (string):
            The directory to save the model to.
        args (NamedTuple):
            A tuple of experiment arguments used (usually provided by HydraConfig).
    """

    # flake8: noqa: C901
    def __init__(self, agents, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opps
        self.random_key = jax.random.PRNGKey(args.seed)
        self.save_dir = save_dir

        def _reshape_opp_dim(x):
            # x: [num_opps, num_envs ...]
            # x: [batch_size, ...]
            batch_size = args.num_envs * args.num_opps
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), x
            )

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        self.ipd_stats = n_player_ipd_visitation
        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )
        self.ipditm_stats = jax.jit(ipditm_stats)
        # VMAP for num opps: we vmap over the rng but not params
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))
        self.num_players = args.num_players
        self.num_shapers = args.num_shapers
        self.num_targets = args.num_players - args.num_shapers
        self.num_outer_steps = self.args.num_outer_steps
        shapers = agents[: self.num_shapers]
        targets = agents[self.num_shapers :]
        # set up agents
        # batch MemoryState not TrainingState
        for agent_idx, shaper_agent in enumerate(shapers):
            agent_arg = f"agent{agent_idx + 1}"
            if OmegaConf.select(args, agent_arg) == "NaiveEx":
                # special case where NaiveEx has a different call signature
                shaper_agent.batch_init = jax.jit(
                    jax.vmap(shaper_agent.make_initial_state)
                )
            else:

                shaper_agent.batch_init = jax.vmap(
                    shaper_agent.make_initial_state,
                    (None, 0),
                    (None, 0),
                )
            shaper_agent.batch_reset = jax.jit(
                jax.vmap(shaper_agent.reset_memory, (0, None), 0),
                static_argnums=1,
            )

            shaper_agent.batch_policy = jax.jit(
                jax.vmap(shaper_agent._policy, (None, 0, 0), (0, None, 0))
            )

        # go through opponents
        for agent_idx, target_agent in enumerate(targets):
            agent_arg = f"agent{agent_idx + self.num_shapers + 1}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) == "NaiveEx":
                target_agent.batch_init = jax.jit(
                    jax.vmap(target_agent.make_initial_state)
                )
            else:
                target_agent.batch_init = jax.vmap(
                    target_agent.make_initial_state, (0, None), 0
                )
            target_agent.batch_policy = jax.jit(jax.vmap(target_agent._policy))
            target_agent.batch_reset = jax.jit(
                jax.vmap(target_agent.reset_memory, (0, None), 0),
                static_argnums=1,
            )
            target_agent.batch_update = jax.jit(
                jax.vmap(target_agent.update, (1, 0, 0, 0), 0)
            )

        for agent_idx, shaper_agent in enumerate(shapers):
            agent_arg = f"agent{agent_idx + 1}"
            if OmegaConf.select(args, agent_arg) != "NaiveEx":
                # NaiveEx requires env first step to init.
                init_hidden = jnp.tile(
                    shaper_agent._mem.hidden, (args.num_opps, 1, 1)
                )
                (
                    shaper_agent._state,
                    shaper_agent._mem,
                ) = shaper_agent.batch_init(
                    shaper_agent._state.random_key, init_hidden
                )

        for agent_idx, target_agent in enumerate(targets):
            agent_arg = f"agent{agent_idx + self.num_shapers + 1}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) != "NaiveEx":
                # NaiveEx requires env first step to init.
                init_hidden = jnp.tile(
                    target_agent._mem.hidden, (args.num_opps, 1, 1)
                )
                (
                    target_agent._state,
                    target_agent._mem,
                ) = target_agent.batch_init(
                    jax.random.split(
                        target_agent._state.random_key, args.num_opps
                    ),
                    init_hidden,
                )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_reward,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ) = carry
            new_targets_mem = [None] * self.num_targets
            new_shapers_mem = [None] * self.num_shapers
            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]
            shapers_actions = []
            targets_actions = []
            for agent_idx, shaper_agent in enumerate(shapers):
                (
                    shaper_action,
                    shapers_state[agent_idx],
                    new_shapers_mem[agent_idx],
                ) = shaper_agent.batch_policy(
                    shapers_state[agent_idx],
                    shapers_obs[agent_idx],
                    shapers_mem[agent_idx],
                )
                shapers_actions.append(shaper_action)
            for agent_idx, target_agent in enumerate(targets):
                (
                    target_action,
                    targets_state[agent_idx],
                    new_targets_mem[agent_idx],
                ) = target_agent.batch_policy(
                    targets_state[agent_idx],
                    targets_obs[agent_idx],
                    targets_mem[agent_idx],
                )
                targets_actions.append(target_action)
            (
                all_agent_next_obs,
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                tuple(shapers_actions + targets_actions),
                env_params,
            )
            shapers_next_obs = all_agent_next_obs[: self.num_shapers]
            targets_next_obs = all_agent_next_obs[self.num_shapers :]
            shapers_reward, targets_reward = (
                all_agent_rewards[: self.num_shapers],
                all_agent_rewards[self.num_shapers :],
            )

            shapers_traj = [
                Sample(
                    shapers_obs[agent_idx],
                    shapers_actions[agent_idx],
                    shapers_reward[agent_idx],
                    new_shapers_mem[agent_idx].extras["log_probs"],
                    new_shapers_mem[agent_idx].extras["values"],
                    done,
                    shapers_mem[agent_idx].hidden,
                )
                for agent_idx in range(self.num_shapers)
            ]
            targets_traj = [
                Sample(
                    targets_obs[agent_idx],
                    targets_actions[agent_idx],
                    targets_rewards[agent_idx],
                    new_targets_mem[agent_idx].extras["log_probs"],
                    new_targets_mem[agent_idx].extras["values"],
                    done,
                    targets_mem[agent_idx].hidden,
                )
                for agent_idx in range(self.num_targets)
            ]
            return (
                rngs,
                tuple(shapers_next_obs),
                tuple(targets_next_obs),
                tuple(shapers_reward),
                tuple(targets_reward),
                shapers_state,
                targets_state,
                new_shapers_mem,
                new_targets_mem,
                env_state,
                env_params,
            ), tuple(shapers_traj + targets_traj)

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
            )
            targets_metrics = [None] * self.num_targets
            (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            for agent_idx, shaper_agent in enumerate(shapers):
                agent_arg = f"agent{agent_idx + 1}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "MFOS":
                    shapers_mem[agent_idx] = shaper_agent.meta_policy(
                        shapers_mem[agent_idx]
                    )

            # update second agent
            targets_traj = trajectories[self.num_shapers :]
            for agent_idx, target_agent in enumerate(targets):
                (
                    targets_state[agent_idx],
                    targets_mem[agent_idx],
                    targets_metrics[agent_idx],
                ) = target_agent.batch_update(
                    targets_traj[agent_idx],
                    targets_obs[agent_idx],
                    targets_state[agent_idx],
                    targets_mem[agent_idx],
                )
            return (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ), (trajectories, targets_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            _shapers_state: List[TrainingState],
            _shapers_mem: List[MemoryState],
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            shapers_obs = obs[: self.num_shapers]
            targets_obs = obs[self.num_shapers :]
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs), dtype=jnp.float32)
            ] * args.num_players
            # Player 1
            for agent_idx, shaper_agent in enumerate(shapers):
                _shapers_mem[agent_idx] = shaper_agent.batch_reset(
                    _shapers_mem[agent_idx], False
                )
            # Other players
            _rng_run, *target_rngs = jax.random.split(
                _rng_run, self.num_players
            )
            targets_mem = [None] * self.num_targets
            targets_state = [None] * self.num_targets
            for agent_idx, target_agent in enumerate(targets):
                # if eg 2 shapers, agent3 is the first non-shaper
                agent_arg = f"agent{agent_idx + 1 + self.num_shapers}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "NaiveEx":
                    (
                        targets_mem[agent_idx],
                        targets_state[agent_idx],
                    ) = target_agent.batch_init(targets_obs[agent_idx])

                elif self.args.env_type in ["meta"]:
                    # meta-experiments - init other agents per trial
                    (
                        targets_state[agent_idx],
                        targets_mem[agent_idx],
                    ) = target_agent.batch_init(
                        jax.random.split(
                            target_rngs[agent_idx], self.num_opps
                        ),
                        target_agent._mem.hidden,
                    )

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    tuple(obs[: self.num_shapers]),
                    tuple(obs[self.num_shapers :]),
                    tuple(rewards[: self.num_shapers]),
                    tuple(rewards[self.num_shapers :]),
                    _shapers_state,
                    targets_state,
                    _shapers_mem,
                    targets_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=self.num_outer_steps,
            )

            (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ) = vals
            trajectories, targets_metrics = stack
            shapers_traj = trajectories[: self.num_shapers]
            targets_traj = trajectories[self.num_shapers :]

            # reset memory
            for agent_idx, shaper_agent in enumerate(shapers):
                shapers_mem[agent_idx] = shaper_agent.batch_reset(
                    shapers_mem[agent_idx], False
                )

            for agent_idx, target_agent in enumerate(targets):
                targets_mem[agent_idx] = target_agent.batch_reset(
                    targets_mem[agent_idx], False
                )
            # Stats
            if args.env_id == "iterated_nplayer_tensor_game":
                total_env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        trajectories[0].observations,
                        num_players=args.num_players,
                    ),
                )
                shapers_rewards = [
                    shapers_traj[shaper_idx].rewards.mean()
                    for shaper_idx in range(self.num_shapers)
                ]

                targets_rewards = [
                    targets_traj[target_idx].rewards.mean()
                    for target_idx in range(self.num_targets)
                ]
            else:
                raise NotImplementedError

            return (
                total_env_stats,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                targets_metrics,
                trajectories,
            )

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        shaper_agents = agents[: self.num_shapers]
        target_agents = agents[self.num_shapers :]
        rng, _ = jax.random.split(self.random_key)

        # get initial state and memory
        shapers_state = []
        shapers_mem = []
        for agent_idx, shaper_agent in enumerate(shaper_agents):
            shapers_state.append(shaper_agent._state)
            shapers_mem.append(shaper_agent._mem)
        targets_state = []
        targets_mem = []
        for agent_idx, target_agent in enumerate(target_agents):
            targets_state.append(target_agent._state)
            targets_mem.append(target_agent._mem)

        for agent_idx, shaper_agent in enumerate(shaper_agents):
            model_path = f"model_path{agent_idx + 1}"
            run_path = f"run_path{agent_idx + 1}"
            if model_path not in self.args:
                raise ValueError(
                    f"Please provide a model path for shaper {agent_idx + 1}"
                )

            wandb.restore(
                name=self.args[model_path],
                run_path=self.args[run_path],
                root=os.getcwd(),
            )
            pretrained_params = load(self.args[model_path])
            shapers_state[agent_idx] = shapers_state[agent_idx]._replace(
                params=pretrained_params
            )

        # run actual loop
        rng, rng_run = jax.random.split(rng, 2)
        # RL Rollout
        (
            env_stats,
            shapers_rewards,
            targets_rewards,
            shapers_state,
            targets_state,
            shapers_mem,
            targets_mem,
            targets_metrics,
            trajectories,
        ) = self.rollout(
            rng_run,
            shapers_state,
            shapers_mem,
            env_params,
        )

        # for stat in env_stats.keys():
        #     print(stat + f": {env_stats[stat].item()}")
        #     print(
        #         f"Shapers Reward Per Timestep: {[float(reward.mean()) for reward in shapers_rewards]}"
        #     )
        #     print(
        #         f"Targets Reward Per Timestep: {[float(reward.mean()) for reward in targets_rewards]}"
        #     )
        # print()

        if watchers:

            list_traj1 = [
                Sample(
                    observations=jax.tree_util.tree_map(
                        lambda x: x[i, ...], trajectories[0].observations
                    ),
                    actions=trajectories[0].actions[i, ...],
                    rewards=trajectories[0].rewards[i, ...],
                    dones=trajectories[0].dones[i, ...],
                    # env_state=None,
                    behavior_log_probs=trajectories[0].behavior_log_probs[
                        i, ...
                    ],
                    behavior_values=trajectories[0].behavior_values[i, ...],
                    hiddens=trajectories[0].hiddens[i, ...],
                )
                for i in range(self.args.num_outer_steps)
            ]

            list_of_env_stats = [
                jax.tree_util.tree_map(
                    lambda x: x.item(),
                    self.ipd_stats(
                        observations=traj.observations,
                        num_players=self.args.num_players,
                    ),
                )
                for traj in list_traj1
            ]
            shaper_traj = trajectories[: self.num_shapers]
            target_traj = trajectories[self.num_shapers :]

            # log agent one
            watchers[0](agents[0])
            # log the inner episodes
            shaper_rewards_log = [
                {
                    f"eval/reward_per_timestep/shaper_{shaper_idx + 1}": float(
                        traj.rewards[i].mean().item()
                    )
                    for (shaper_idx, traj) in enumerate(shaper_traj)
                }
                for i in range(len(list_of_env_stats))
            ]
            target_rewards_log = [
                {
                    f"eval/reward_per_timestep/target_{target_idx + 1}": float(
                        traj.rewards[i].mean().item()
                    )
                    for (target_idx, traj) in enumerate(target_traj)
                }
                for i in range(len(list_of_env_stats))
            ]

            # log avg reward for players combined
            global_welfare_log = [
                {
                    f"eval/global_welfare_per_timestep": float(
                        sum(
                            traj.rewards[i].mean().item()
                            for traj in trajectories
                        )
                    )
                    / len(trajectories)
                }
                for i in range(len(list_of_env_stats))
            ]
            shaper_welfare_log = [
                {
                    f"eval/shaper_welfare_per_timestep": float(
                        sum(
                            traj.rewards[i].mean().item()
                            for traj in shaper_traj
                        )
                    )
                    / len(shaper_traj)
                }
                for i in range(len(list_of_env_stats))
            ]
            target_welfare_log = [
                {
                    f"eval/target_welfare_per_timestep": float(
                        sum(
                            traj.rewards[i].mean().item()
                            for traj in target_traj
                        )
                    )
                    / len(target_traj)
                }
                for i in range(len(list_of_env_stats))
            ]

            for i in range(len(list_of_env_stats)):
                wandb.log(
                    {
                        "train_iteration": i,
                    }
                    | list_of_env_stats[i]
                    | shaper_rewards_log[i]
                    | target_rewards_log[i]
                    | shaper_welfare_log[i]
                    | target_welfare_log[i]
                    | global_welfare_log[i]
                )
            shaper_rewards_log = {
                f"eval/meta_reward/shaper{idx + 1}": float(rew.mean().item())
                for (idx, rew) in enumerate(shapers_rewards)
            }
            target_rewards_log = {
                f"eval/meta_reward/target{idx + 1}": float(rew.mean().item())
                for (idx, rew) in enumerate(targets_rewards)
            }

            wandb.log(
                {
                    "episodes": 1,
                }
                | shaper_rewards_log
                | target_rewards_log
            )

        return agents
