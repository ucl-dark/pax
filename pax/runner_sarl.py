import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.watchers import cg_visitation, ipd_visitation
from pax.utils import MemoryState, TrainingState, save

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

@jax.jit
def reduce_outer_traj(traj: Sample) -> Sample:
    """Used to collapse lax.scan outputs dims"""
    # x: [outer_loop, inner_loop, num_envs ...]
    # x: [timestep, batch_size, ...]
    num_timesteps = traj.observations.shape[0] * traj.observations.shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_timesteps,) + x.shape[2:]),
        traj,
    )


class SARLRunner:
    """Holds the runner's state."""

    def __init__(self, agents, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.random_key = jax.random.PRNGKey(args.seed)
        self.save_dir = save_dir

        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)
        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )

        self.split = jax.vmap(jax.random.split, (0, None))
        num_outer_steps = (
            1
            if self.args.env_type == "sequential"
            else self.args.num_steps // self.args.num_inner_steps
        )

        agent1 = agents

        # set up agents
        if args.agent1 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent1.batch_init = jax.jit(jax.vmap(agent1.make_initial_state))
        else:
            # batch MemoryState not TrainingState
            agent1.batch_init = agent1.make_initial_state 
        
        agent1.batch_reset = jax.jit(
            agent1.reset_memory, static_argnums=1
        )

        agent1.batch_policy = jax.jit(agent1._policy)


        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent1._mem.hidden, (1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )


        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                obs,
                rewards,
                a1_state,
                a1_mem,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, 0, :]
            # a1_rng = rngs[:, 1, :]
            # a2_rng = rngs[:, 2, :]
            rngs = rngs[:, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
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
                rewards,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )   

            return (
                rngs,
                next_obs, # next_obs
                rewards,
                a1_state,
                new_a1_mem,
                env_state,
                env_params,
            ), traj1

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
                r1,
                a1_state,
                a1_mem,
                env_state,
                env_params,
            ) = vals

            return (
                rngs,
                obs1,
                r1,
                a1_state,
                a1_mem,
                env_state,
                env_params,
            ), trajectories

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
            rewards = jnp.zeros((args.num_envs))

            # Player 1
            _a1_mem = agent1.batch_reset(_a1_mem, False)

            # run trials
            vals, traj = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    obs,
                    rewards,
                    _a1_state,
                    _a1_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=num_outer_steps,
            )

            (
                rngs,
                obs1,
                r1,
                a1_state,
                a1_mem,
                env_state,
                env_params,
            ) = vals

            # update outer agent
            a1_state, _, a1_metrics = agent1.update(
                reduce_outer_traj(traj),
                obs1,
                r1,
                jnp.ones_like(r1, dtype=jnp.bool_),
                a1_state,
                a1_mem,
            )

            # reset memory
            a1_mem = agent1.batch_reset(a1_mem, False)

            # Stats
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x,
                    self.cg_stats(env_state),
                )

                rewards = traj.rewards.sum(axis=1).mean()

            elif args.env_id in [
                "ipd",
            ]:
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj.observations,
                        traj.actions,
                        obs1,
                    ),
                )
                rewards = traj.rewards.mean()
            else:
                #TODO: add stats @timon
                rewards = traj.rewards.mean()
                env_stats = {}

            return (
                env_stats,
                rewards,
                a1_state,
                a1_mem,
                a1_metrics,
            )

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env, env_params, agents, num_iters, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1= agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem

        num_iters = max(
            int(num_iters / (self.args.num_envs)), 1
        )
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
            ) = self.rollout(
                rng_run, a1_state, a1_mem, env_params
            )

            if self.args.save and i % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            # logging
            self.train_episodes += 1
            if i % log_interval == 0:
                print(f"Episode {i}")

                print(f"Env Stats: {env_stats}")
                print(
                    f"Total Episode Reward: {float(rewards_1.mean())}"
                )
                print()

                if watchers:
                    # metrics [outer_timesteps]
                    flattened_metrics_1 = jax.tree_util.tree_map(
                        lambda x: jnp.mean(x), a1_metrics
                    )
                    agent1._logger.metrics = (
                        agent1._logger.metrics | flattened_metrics_1
                    )

                    for watcher, agent in zip(watchers, agents):
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

        agents._state = a1_state
        return agents

