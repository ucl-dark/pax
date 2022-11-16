import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.watchers import cg_visitation, ipd_visitation
from pax.utils import MemoryState, TrainingState, save

# from jax.config import config
# config.update('jax_disable_jit', True)

MAX_WANDB_CALLS = 1000000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class SARLRunner:
    """Holds the runner's state."""

    def __init__(self, agent, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.random_key = jax.random.PRNGKey(args.seed)
        self.save_dir = save_dir

        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.random.split, (0, None))
        # set up agent
        if args.agent1 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent.batch_init = jax.jit(jax.vmap(agent.make_initial_state))
        else:
            # batch MemoryState not TrainingState
            agent.batch_init = jax.jit(agent.make_initial_state)

        agent.batch_reset = jax.jit(agent.reset_memory, static_argnums=1)

        agent.batch_policy = jax.jit(agent._policy)

        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent._mem.hidden, (1))
            agent._state, agent._mem = agent.batch_init(
                agent._state.random_key, init_hidden
            )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                obs,
                a1_state,
                a1_mem,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            # import pdb; pdb.set_trace()
            rngs = self.split(rngs, 2)
            env_rng = rngs[:, 0, :]
            # a1_rng = rngs[:, 1, :]
            # a2_rng = rngs[:, 2, :]
            rngs = rngs[:, 1, :]

            a1, a1_state, new_a1_mem = agent.batch_policy(
                a1_state,
                obs,
                a1_mem,
            )

            next_obs, env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                a1,
                env_params,
            )

            traj1 = Sample(
                obs,
                a1,
                rewards * jnp.logical_not(done),
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )

            return (
                rngs,
                next_obs,  # next_obs
                a1_state,
                new_a1_mem,
                env_state,
                env_params,
            ), traj1

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
            ).reshape((args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            _a1_mem = agent.batch_reset(_a1_mem, False)

            # run trials
            vals, traj = jax.lax.scan(
                _inner_rollout,
                (   
                    rngs,
                    obs,
                    _a1_state,
                    _a1_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=args.num_steps,
            )

            (
                rngs,
                obs,
                _a1_state,
                _a1_mem,
                env_state,
                env_params,
            ) = vals

            # update outer agent
            _a1_state, _, _a1_metrics = agent.update(
                traj,
                obs,
                _a1_state,
                _a1_mem,
            )

            # reset memory
            _a1_mem = agent.batch_reset(_a1_mem, False)

            # Stats
            rewards = jnp.sum(traj.rewards)/(jnp.sum(traj.dones)+1e-8)
            env_stats = {}

            return (
                env_stats,
                rewards,
                _a1_state,
                _a1_mem,
                _a1_metrics,
            )

        self.rollout = _rollout
        # self.rollout = jax.jit(_rollout)

    def run_loop(self, env, env_params, agent, num_iters, watcher):
        """Run training of agent in environment"""
        print("Training")
        print("-----------------------")
        agent = agent
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent._state, agent._mem

        num_iters = max(int(num_iters / (self.args.num_envs)), 1)
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)

        print(f"Log Interval {log_interval}")
        print(f"Running for total iterations: {num_iters}")
        # run actual loop
        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            # RL Rollout
            (
                env_stats,
                rewards_1,
                a1_state,
                a1_mem,
                a1_metrics,
            ) = self.rollout(rng_run, a1_state, a1_mem, env_params)

            if self.args.save and i % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watcher:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            # logging
            self.train_episodes += 1
            if num_iters % log_interval == 0:
                print(f"Episode {i}")

                print(f"Env Stats: {env_stats}")
                print(f"Total Episode Reward: {float(rewards_1.mean())}")
                print()

                if watcher:
                    # metrics [outer_timesteps]
                    flattened_metrics_1 = jax.tree_util.tree_map(
                        lambda x: jnp.mean(x), a1_metrics
                    )
                    agent._logger.metrics = (
                        agent._logger.metrics | flattened_metrics_1
                    )

                    watcher(agent)
                    wandb.log(
                        {
                            "episodes": self.train_episodes,
                            "train/episode_reward/player_1": float(
                                rewards_1.mean()
                            ),
                        }
                        | env_stats,
                    )

        agent._state = a1_state
        return agent