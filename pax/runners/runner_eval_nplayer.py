import os
import time
from typing import Any, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import wandb
from pax.utils import MemoryState, TrainingState, save, load
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


class NPlayerEvalRunner:
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
        self.num_outer_steps = self.args.num_outer_steps
        agent1, *other_agents = agents
        # set up agents
        if args.agent1 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent1.batch_init = jax.jit(jax.vmap(agent1.make_initial_state))
        else:
            # batch MemoryState not TrainingState
            agent1.batch_init = jax.vmap(
                agent1.make_initial_state,
                (None, 0),
                (None, 0),
            )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )

        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # go through opponents, we start with agent2
        for agent_idx, non_first_agent in enumerate(other_agents):
            agent_arg = f"agent{agent_idx+2}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) == "NaiveEx":
                non_first_agent.batch_init = jax.jit(
                    jax.vmap(non_first_agent.make_initial_state)
                )
            else:
                non_first_agent.batch_init = jax.vmap(
                    non_first_agent.make_initial_state, (0, None), 0
                )
            non_first_agent.batch_policy = jax.jit(
                jax.vmap(non_first_agent._policy)
            )
            non_first_agent.batch_reset = jax.jit(
                jax.vmap(non_first_agent.reset_memory, (0, None), 0),
                static_argnums=1,
            )
            non_first_agent.batch_update = jax.jit(
                jax.vmap(non_first_agent.update, (1, 0, 0, 0), 0)
            )

        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )

        for agent_idx, non_first_agent in enumerate(other_agents):
            agent_arg = f"agent{agent_idx+2}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) != "NaiveEx":
                # NaiveEx requires env first step to init.
                init_hidden = jnp.tile(
                    non_first_agent._mem.hidden, (args.num_opps, 1, 1)
                )
                (
                    non_first_agent._state,
                    non_first_agent._mem,
                ) = non_first_agent.batch_init(
                    jax.random.split(
                        non_first_agent._state.random_key, args.num_opps
                    ),
                    init_hidden,
                )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = carry
            new_other_agent_mem = [None] * len(other_agents)
            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]
            actions = []
            (
                first_action,
                first_agent_state,
                new_first_agent_mem,
            ) = agent1.batch_policy(
                first_agent_state,
                first_agent_obs,
                first_agent_mem,
            )

            actions.append(first_action)
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    non_first_action,
                    other_agent_state[agent_idx],
                    new_other_agent_mem[agent_idx],
                ) = non_first_agent.batch_policy(
                    other_agent_state[agent_idx],
                    other_agent_obs[agent_idx],
                    other_agent_mem[agent_idx],
                )
                actions.append(non_first_action)
            (
                all_agent_next_obs,
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                actions,
                env_params,
            )
            first_agent_next_obs, *other_agent_next_obs = all_agent_next_obs
            first_agent_reward, *other_agent_rewards = all_agent_rewards

            traj1 = Sample(
                first_agent_next_obs,
                first_action,
                first_agent_reward,
                new_first_agent_mem.extras["log_probs"],
                new_first_agent_mem.extras["values"],
                done,
                first_agent_mem.hidden,
            )
            other_traj = [
                Sample(
                    other_agent_next_obs[agent_idx],
                    actions[agent_idx + 1],
                    other_agent_rewards[agent_idx],
                    new_other_agent_mem[agent_idx].extras["log_probs"],
                    new_other_agent_mem[agent_idx].extras["values"],
                    done,
                    other_agent_mem[agent_idx].hidden,
                )
                for agent_idx in range(len(other_agents))
            ]
            return (
                rngs,
                first_agent_next_obs,
                tuple(other_agent_next_obs),
                first_agent_reward,
                tuple(other_agent_rewards),
                first_agent_state,
                other_agent_state,
                new_first_agent_mem,
                new_other_agent_mem,
                env_state,
                env_params,
            ), (traj1, *other_traj)

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
            )
            other_agent_metrics = [None] * len(other_agents)
            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                first_agent_mem = agent1.meta_policy(first_agent_mem)

            # update second agent
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                    other_agent_metrics[agent_idx],
                ) = non_first_agent.batch_update(
                    trajectories[agent_idx + 1],
                    other_agent_obs[agent_idx],
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                )
            return (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ), (trajectories, other_agent_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            first_agent_state: TrainingState,
            first_agent_mem: MemoryState,
            other_agent_state: List[TrainingState],
            other_agent_mem: List[MemoryState],
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs), dtype=jnp.float32)
            ] * args.num_players
            # Player 1
            first_agent_mem = agent1.batch_reset(first_agent_mem, False)
            # Other players
            _rng_run, other_agent_rng = jax.random.split(_rng_run, 2)
            for agent_idx, non_first_agent in enumerate(other_agents):
                # indexing starts at 2 for args
                agent_arg = f"agent{agent_idx+2}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "NaiveEx":
                    (
                        other_agent_mem[agent_idx],
                        other_agent_state[agent_idx],
                    ) = non_first_agent.batch_init(obs[agent_idx + 1])

                elif self.args.env_type in ["meta"]:
                    # meta-experiments - init other agents per trial
                    (
                        other_agent_state[agent_idx],
                        other_agent_mem[agent_idx],
                    ) = non_first_agent.batch_init(
                        jax.random.split(other_agent_rng, self.num_opps),
                        non_first_agent._mem.hidden,
                    )
                    _rng_run, other_agent_rng = jax.random.split(_rng_run, 2)

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    obs[0],
                    tuple(obs[1:]),
                    rewards[0],
                    tuple(rewards[1:]),
                    first_agent_state,
                    other_agent_state,
                    first_agent_mem,
                    other_agent_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=self.num_outer_steps,
            )

            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = vals
            trajectories, other_agent_metrics = stack

            # reset memory
            first_agent_mem = agent1.batch_reset(first_agent_mem, False)
            for agent_idx, non_first_agent in enumerate(other_agents):
                other_agent_mem[agent_idx] = non_first_agent.batch_reset(
                    other_agent_mem[agent_idx], False
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
                total_rewards = [traj.rewards.mean() for traj in trajectories]
            else:
                total_env_stats = {}
                total_rewards = [traj.rewards.mean() for traj in trajectories]

            return (
                total_env_stats,
                total_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                other_agent_metrics,
                trajectories,
            )

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, *other_agents = agents
        rng, _ = jax.random.split(self.random_key)

        first_agent_state, first_agent_mem = agent1._state, agent1._mem
        other_agent_mem = [None] * len(other_agents)
        other_agent_state = [None] * len(other_agents)

        for agent_idx, non_first_agent in enumerate(other_agents):
            other_agent_state[agent_idx], other_agent_mem[agent_idx] = (
                non_first_agent._state,
                non_first_agent._mem,
            )

        wandb.restore(
            name=self.args.model_path1,
            run_path=self.args.run_path1,
            root=os.getcwd(),
        )

        pretrained_params = load(self.args.model_path1)
        first_agent_state = first_agent_state._replace(
            params=pretrained_params
        )

        # run actual loop
        rng, rng_run = jax.random.split(rng, 2)
        # RL Rollout
        (
            env_stats,
            total_rewards,
            first_agent_state,
            other_agent_state,
            first_agent_mem,
            other_agent_mem,
            other_agent_metrics,
            trajectories,
        ) = self.rollout(
            rng_run,
            first_agent_state,
            first_agent_mem,
            other_agent_state,
            other_agent_mem,
            env_params,
        )

        for stat in env_stats.keys():
            print(stat + f": {env_stats[stat].item()}")
        print(
            f"Total Episode Reward: {[float(rew.mean()) for rew in total_rewards]}"
        )
        print()

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

            # log agent one
            watchers[0](agents[0])
            # log the inner episodes
            rewards_log = [
                {
                    f"eval/reward_per_timestep/player_{agent_idx+1}": float(
                        traj.rewards[i].mean().item()
                    )
                    for (agent_idx, traj) in enumerate(trajectories)
                }
                for i in range(len(list_of_env_stats))
            ]

            for i in range(len(list_of_env_stats)):
                wandb.log(
                    {
                        "train_iteration": i,
                    }
                    | list_of_env_stats[i]
                    | rewards_log[i]
                )
            total_rewards_log = {
                f"eval/meta_reward/player_{idx+1}": float(rew.mean().item())
                for (idx, rew) in enumerate(total_rewards)
            }
            wandb.log(
                {
                    "episodes": 1,
                }
                | total_rewards_log
            )

        return agents
