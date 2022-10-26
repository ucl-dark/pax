import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.watchers import cg_visitation, ipd_visitation
from pax.utils import load

MAX_WANDB_CALLS = 10000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class EvalRunner:
    """Evaluation runner"""

    def __init__(self, args):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opps
        self.random_key = jax.random.PRNGKey(args.seed)
        self.run_path = args.run_path
        self.model_path = args.model_path
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)

    def run_loop(self, env, agents, num_episodes, watchers):
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

            final_t2 = t2._replace(step_type=2 * jnp.ones_like(t2.step_type))

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

        if watchers:
            wandb.restore(
                name=self.model_path, run_path=self.run_path, root=os.getcwd()
            )
        pretrained_params = load(self.model_path)
        a1_state = a1_state._replace(params=pretrained_params)

        num_iters = max(int(num_episodes / (env.num_envs * self.num_opps)), 1)
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Log Interval {log_interval}")
        # run actual loop
        for i in range(num_episodes):
            rng, rng_run = jax.random.split(rng)
            t_init, env_state = env.runner_reset(
                (self.num_opps, env.num_envs), rng_run
            )

            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(t_init[1])

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.outer_ep_length,
            )

            t1, t2, _, a1_mem, a2_state, a2_mem, env_state = vals
            traj_1, traj_2, a2_metrics = stack

            # reset second agent memory
            a2_mem = agent2.batch_reset(a2_mem, False)

            # logging
            self.train_episodes += 1
            if i % log_interval == 0:
                print(f"Episode {i}")
                if self.args.env_id == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.cg_stats(env_state),
                    )
                    rewards_0 = traj_1.rewards.sum(axis=1).mean()
                    rewards_1 = traj_2.rewards.sum(axis=1).mean()

                elif self.args.env_type in [
                    "meta",
                    "sequential",
                ]:
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.ipd_stats(
                            traj_1.observations,
                            traj_1.actions,
                            t1.observation,
                        ),
                    )
                    rewards_0 = traj_1.rewards.mean()
                    rewards_1 = traj_2.rewards.mean()

                else:
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
