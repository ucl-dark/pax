import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

import wandb
from pax.utils import MemoryState, TrainingState, save, load
from pax.watchers import ipditm_stats, tensor_ipd_visitation

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


class TensorEvalRunner:
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
        self.tensor_ipd_stats = jax.jit(tensor_ipd_visitation)
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

        agent1, agent2, agent3 = agents

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

        # batch all for Agent2
        if args.agent2 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        # batch all for Agent3
        if args.agent3 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent3.batch_init = jax.jit(jax.vmap(agent3.make_initial_state))
        else:
            agent3.batch_init = jax.vmap(
                agent3.make_initial_state, (0, None), 0
            )
        agent3.batch_policy = jax.jit(jax.vmap(agent3._policy))
        agent3.batch_reset = jax.jit(
            jax.vmap(agent3.reset_memory, (0, None), 0), static_argnums=1
        )
        agent3.batch_update = jax.jit(jax.vmap(agent3.update, (1, 0, 0, 0), 0))

        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )

        if args.agent2 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
            agent2._state, agent2._mem = agent2.batch_init(
                jax.random.split(agent2._state.random_key, args.num_opps),
                init_hidden,
            )
        if args.agent3 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent3._mem.hidden, (args.num_opps, 1, 1))
            agent3._state, agent3._mem = agent3.batch_init(
                jax.random.split(agent3._state.random_key, args.num_opps),
                init_hidden,
            )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                obs1,
                obs2,
                obs3,
                r1,
                r2,
                r3,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                a3_state,
                a3_mem,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                obs1,
                a1_mem,
            )

            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                obs2,
                a2_mem,
            )
            a3, a3_state, new_a3_mem = agent3.batch_policy(
                a3_state,
                obs3,
                a3_mem,
            )
            (
                (next_obs1, next_obs2, next_obs3),
                env_state,
                rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                (a1, a2, a3),
                env_params,
            )

            traj1 = Sample(
                obs1,
                a1,
                rewards[0],
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )
            traj2 = Sample(
                obs2,
                a2,
                rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
            )
            traj3 = Sample(
                obs3,
                a3,
                rewards[2],
                new_a3_mem.extras["log_probs"],
                new_a3_mem.extras["values"],
                done,
                a3_mem.hidden,
            )
            return (
                rngs,
                next_obs1,
                next_obs2,
                next_obs3,
                rewards[0],
                rewards[1],
                rewards[2],
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                a3_state,
                new_a3_mem,
                env_state,
                env_params,
            ), (
                traj1,
                traj2,
                traj3,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
            )
            (
                rngs,
                obs1,
                obs2,
                obs3,
                r1,
                r2,
                r3,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                a3_state,
                a3_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                a1_mem = agent1.meta_policy(a1_mem)

            # update second agent
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1],
                obs2,
                a2_state,
                a2_mem,
            )
            a3_state, a3_mem, a3_metrics = agent3.batch_update(
                trajectories[2],
                obs3,
                a3_state,
                a3_mem,
            )
            return (
                rngs,
                obs1,
                obs2,
                obs3,
                r1,
                r2,
                r3,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                a3_state,
                a3_mem,
                env_state,
                env_params,
            ), (*trajectories, a2_metrics, a3_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _a2_mem: MemoryState,
            _a3_state: TrainingState,
            _a3_mem: MemoryState,
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
            ]
            # Player 1
            _a1_mem = agent1.batch_reset(_a1_mem, False)

            # Player 2
            if args.agent1 == "NaiveEx":
                _a1_state, _a1_mem = agent1.batch_init(obs[0])

            if args.agent2 == "NaiveEx":
                _a2_state, _a2_mem = agent2.batch_init(obs[1])
            if args.agent3 == "NaiveEx":
                _a3_state, _a3_mem = agent3.batch_init(obs[2])

            elif self.args.env_type in ["meta"]:
                _rng_run, agent2_rng = jax.random.split(_rng_run, 2)
                _rng_run, agent3_rng = jax.random.split(_rng_run, 2)
                # meta-experiments - init 2nd agent per trial
                _a2_state, _a2_mem = agent2.batch_init(
                    jax.random.split(agent2_rng, self.num_opps), _a2_mem.hidden
                )
                # meta-experiments - init 3nd agent per trial
                _a3_state, _a3_mem = agent3.batch_init(
                    jax.random.split(agent3_rng, self.num_opps), _a3_mem.hidden
                )
            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    _a1_state,
                    _a1_mem,
                    _a2_state,
                    _a2_mem,
                    _a3_state,
                    _a3_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=self.args.num_outer_steps,
            )

            (
                rngs,
                obs1,
                obs2,
                obs3,
                r1,
                r2,
                r3,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                a3_state,
                a3_mem,
                env_state,
                env_params,
            ) = vals
            traj_1, traj_2, traj_3, a2_metrics, a3_metrics = stack

            # reset memory
            a1_mem = agent1.batch_reset(a1_mem, False)
            a2_mem = agent2.batch_reset(a2_mem, False)
            a3_mem = agent3.batch_reset(a3_mem, False)
            # Stats
            if args.env_id == "iterated_tensor_game":
                total_env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.tensor_ipd_stats(
                        traj_1.observations,
                    ),
                )
                total_rewards_1 = traj_1.rewards.mean()
                total_rewards_2 = traj_2.rewards.mean()
                total_rewards_3 = traj_3.rewards.mean()
            else:
                total_env_stats = {}
                total_rewards_1 = traj_1.rewards.mean()
                total_rewards_2 = traj_2.rewards.mean()
                total_rewards_3 = traj_3.rewards.mean()

            return (
                total_env_stats,
                total_rewards_1,
                total_rewards_2,
                total_rewards_3,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                a2_metrics,
                a3_state,
                a3_mem,
                a3_metrics,
                traj_1,
                traj_2,
                traj_3,
            )

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2, agent3 = agents
        rng, _ = jax.random.split(self.random_key)
        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem
        a3_state, a3_mem = agent3._state, agent3._mem

        wandb.restore(
            name=self.args.model_path1,
            run_path=self.args.run_path1,
            root=os.getcwd(),
        )

        pretrained_params = load(self.args.model_path1)
        a1_state = a1_state._replace(params=pretrained_params)

        # run actual loop
        rng, rng_run = jax.random.split(rng, 2)
        # RL Rollout
        (
            env_stats,
            total_rewards_1,
            total_rewards_2,
            total_rewards_3,
            a1_state,
            a1_mem,
            a2_state,
            a2_mem,
            a2_metrics,
            a3_state,
            a3_mem,
            a3_metrics,
            traj1,
            traj2,
            traj3,
        ) = self.rollout(
            rng_run,
            a1_state,
            a1_mem,
            a2_state,
            a2_mem,
            a3_state,
            a3_mem,
            env_params,
        )

        for stat in env_stats.keys():
            print(stat + f": {env_stats[stat].item()}")
        print(
            f"Total Episode Reward: {float(total_rewards_1.mean()), float(total_rewards_2.mean()), float(total_rewards_3.mean())}"
        )
        print()

        if watchers:

            list_traj1 = [
                Sample(
                    observations=jax.tree_util.tree_map(
                        lambda x: x[i, ...], traj1.observations
                    ),
                    actions=traj1.actions[i, ...],
                    rewards=traj1.rewards[i, ...],
                    dones=traj1.dones[i, ...],
                    # env_state=None,
                    behavior_log_probs=traj1.behavior_log_probs[i, ...],
                    behavior_values=traj1.behavior_values[i, ...],
                    hiddens=traj1.hiddens[i, ...],
                )
                for i in range(self.args.num_outer_steps)
            ]

            list_of_env_stats = [
                jax.tree_util.tree_map(
                    lambda x: x.item(),
                    self.tensor_ipd_stats(
                        traj.observations,
                    ),
                )
                for traj in list_traj1
            ]

            # log agent one
            watchers[0](agents[0])
            # log the inner episodes
            for i in range(len(list_of_env_stats)):
                wandb.log(
                    {
                        "train_iteration": i,
                        "eval/reward_per_timestep/player_1": float(
                            traj1.rewards[i].mean().item()
                        ),
                        "eval/reward_per_timestep/player_2": float(
                            traj2.rewards[i].mean().item()
                        ),
                        "eval/reward_per_timestep/player_3": float(
                            traj3.rewards[i].mean().item()
                        ),
                    }
                    | list_of_env_stats[i]
                )
            wandb.log(
                {
                    "episodes": 1,
                    "eval/meta_reward/player_1": float(
                        total_rewards_1.mean().item()
                    ),
                    "eval/meta_reward/player_2": float(
                        total_rewards_2.mean().item()
                    ),
                    "eval/meta_reward/player_3": float(
                        total_rewards_3.mean().item()
                    ),
                }
            )

        return agents
