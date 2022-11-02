import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.watchers import cg_visitation, ipd_visitation
from pax.utils import save

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


class RLRunner:
    """Holds the runner's state."""

    def __init__(self, env, save_dir, args):
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
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)
        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )

        # VMAP for num opps: we vmap over the rng but not params
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

    def run_loop(self, env, env_params, agents, num_episodes, watchers):
        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
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
            (next_obs1, next_obs2), env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                (a1, a2),
                env_params,
            )

            if self.args.agent1 == "MFOS":
                traj1 = MFOSSample(
                    obs1,
                    a1,
                    rewards[0],
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    done,
                    a1_mem.hidden,
                    a1_mem.th,
                )
            else:
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
            return (
                rngs,
                next_obs1,
                next_obs2,
                rewards[0],
                rewards[1],
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
                env_params,
            ), (
                traj1,
                traj2,
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
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if self.args.agent1 == "MFOS":
                a1_mem = agent1.meta_policy(a1_mem)

            # update second agent
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1],
                obs2,
                r2,
                jnp.ones_like(r2, dtype=jnp.bool_),
                a2_state,
                a2_mem,
            )
            return (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ), (*trajectories, a2_metrics)

        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem
        num_iters = max(
            int(num_episodes / (self.args.num_envs * self.num_opps)), 1
        )

        num_outer_steps = (
            1
            if self.args.env_type == "sequential"
            else self.args.num_steps // self.args.num_inner_steps
        )
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)

        print(f"Log Interval {log_interval}")
        # RNG are the same for num_opps but different for num_envs
        rngs = jnp.concatenate(
            [jax.random.split(rng, self.args.num_envs)] * self.args.num_opps
        ).reshape((self.args.num_opps, self.args.num_envs, -1))

        # run actual loop
        for i in range(num_episodes):
            obs, env_state = env.reset(rngs, env_params)
            rewards = [
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
            ]

            if self.args.agent1 == "NaiveEx":
                a1_state, a1_mem = agent1.batch_init(obs[0])

            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(obs[1])

            elif self.args.env_type in ["meta", "infinite"]:
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps), a2_mem.hidden
                )
            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    a1_state,
                    a1_mem,
                    a2_state,
                    a2_mem,
                    env_state,
                    env_params,
                ),
                None,
                length=num_outer_steps,
            )

            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # update outer agent
            a1_state, _, _ = agent1.update(
                reduce_outer_traj(traj_1),
                self.reduce_opp_dim(obs1),
                self.reduce_opp_dim(r1),
                jnp.ones_like(self.reduce_opp_dim(r1), dtype=jnp.bool_),
                a1_state,
                self.reduce_opp_dim(a1_mem),
            )

            # reset memory
            a1_mem = agent1.batch_reset(a1_mem, False)
            a2_mem = agent2.batch_reset(a2_mem, False)

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

                if self.args.env_id == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.mean().item(),
                        self.cg_stats(env_state),
                    )
                    rewards_0 = traj_1.rewards.sum(axis=1).mean()
                    rewards_1 = traj_2.rewards.sum(axis=1).mean()

                elif self.args.env_id in [
                    "ipd",
                ]:
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.ipd_stats(
                            traj_1.observations,
                            traj_1.actions,
                            obs1,
                        ),
                    )
                    rewards_0 = traj_1.rewards.mean()
                    rewards_1 = traj_2.rewards.mean()

                else:
                    rewards_0 = traj_1.rewards.mean()
                    rewards_1 = traj_2.rewards.mean()
                    env_stats = {}

                print(f"Env Stats: {env_stats}")
                print(
                    f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
                )
                print()

                if watchers:
                    # metrics [outer_timesteps, num_opps]
                    flattened_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flattened_metrics
                    )

                    agents.log(watchers)
                    wandb.log(
                        {
                            "episodes": self.train_episodes,
                            "train/episode_reward/player_1": float(
                                rewards_0.mean()
                            ),
                            "train/episode_reward/player_2": float(
                                rewards_1.mean()
                            ),
                        }
                        | env_stats,
                    )

        agents.agents[0]._state = a1_state
        agents.agents[1]._state = a2_state
        return agents
