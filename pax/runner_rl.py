import os
import time
from typing import List, NamedTuple

import jax
import jax.numpy as jnp
import wandb
from dm_env import TimeStep

from pax.utils import save

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


class Runner:
    """Holds the runner's state."""

    def __init__(self, args, save_dir):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
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

        def _state_visitation(traj: Sample, final_t: TimeStep) -> List:
            # obs [num_outer_steps, num_inner_steps, num_opps, num_envs, ...]
            # final_t [num_opps, num_envs, ...]
            num_timesteps = (
                traj.observations.shape[0] * traj.observations.shape[1]
            )
            # obs = [0, 1, 2, 3, 4], a = [0, 1]
            # combine = [0, .... 9]
            state_actions = (
                2 * jnp.argmax(traj.observations, axis=-1) + traj.actions
            )
            state_actions = jnp.reshape(
                state_actions,
                (num_timesteps,) + state_actions.shape[2:],
            )
            # assume final step taken is cooperate
            final_obs = jax.lax.expand_dims(
                2 * jnp.argmax(final_t.observation, axis=-1), [0]
            )
            state_actions = jnp.append(state_actions, final_obs, axis=0)
            return jnp.bincount(state_actions.flatten(), length=10)

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        # self.state_visitation = jax.jit(_state_visitation)
        self.state_visitation = _state_visitation

    def train_loop(self, env, agents, num_episodes, watchers):
        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                t1.observation,
                a1_mem,
            )

            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                t2.observation,
                a2_mem,
            )
            (tprime_1, tprime_2), env_state = env.batch_step(
                (a1, a2),
                env_state,
            )

            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                tprime_1.last(),
                a1_mem.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                tprime_2.last(),
                a2_mem.hidden,
            )
            return (
                tprime_1,
                tprime_2,
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            t1, t2, a1_state, a1_mem, a2_state, a2_memory, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=env.inner_episode_length,
            )

            # update second agent
            t1, t2, a1_state, a1_mem, a2_state, a2_memory, env_state = vals

            final_t2 = t2._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )

            a2_state, a2_memory, a2_metrics = agent2.batch_update(
                trajectories[1], final_t2, a2_state, a2_memory
            )

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                a2_memory,
                env_state,
            ), (*trajectories, a2_metrics)

        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem
        num_iters = max(int(num_episodes / (env.num_envs * self.num_opps)), 1)
        log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        print(f"Log Interval {log_interval}")
        print(f"Save directory: {self.save_dir}")

        for i in range(
            0, max(int(num_episodes / (env.num_envs * self.num_opps)), 1)
        ):
            rng, rng_run = jax.random.split(rng)
            t_init, env_state = env.runner_reset(
                (self.num_opps, env.num_envs), rng_run
            )

            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(t_init[1])

            elif self.args.env_type in ["meta", "infinite"]:
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps), a2_mem.hidden
                )

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            traj_1, traj_2, a2_metrics = stack
            # update outer agent
            final_t1 = vals[0]._replace(
                step_type=2 * jnp.ones_like(vals[0].step_type)
            )
            a1_state = vals[2]
            a1_mem = vals[3]

            a1_state, _, _ = agent1.update(
                reduce_outer_traj(traj_1),
                self.reduce_opp_dim(final_t1),
                a1_state,
                self.reduce_opp_dim(a1_mem),
            )
            a1_mem = agent1.batch_reset(a1_mem, False)

            # update second agent
            a2_state, a2_mem = vals[4], vals[5]

            # logging
            self.train_episodes += 1
            rewards_0 = stack[0].rewards.mean()
            rewards_1 = stack[1].rewards.mean()

            if self.args.save and i % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            if i % log_interval == 0:
                print(f"Iteration {i}")
                print(
                    f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
                )

                visits = self.state_visitation(traj_1, final_t1)
                states = visits.reshape((int(visits.shape[0] / 2), 2)).sum(
                    axis=1
                )
                print(f"State Frequency: {states}")
                action_probs = visits[::2] / states
                action_probs = jnp.nan_to_num(action_probs)
                print(f"Cooperation Probability: {action_probs}")

                if watchers:
                    # metrics [outer_timesteps, num_opps]
                    flattened_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flattened_metrics
                    )

                    agent1._logger.metrics = (
                        agent1._logger.metrics | flattened_metrics
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
                        },
                    )
                print()

        agents.agents[0]._state = a1_state
        agents.agents[1]._state = a2_state
        return agents
