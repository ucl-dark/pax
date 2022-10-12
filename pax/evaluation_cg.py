from datetime import datetime
import os
import time
from typing import List, NamedTuple

from dm_env import TimeStep
import jax
import jax.numpy as jnp
import wandb

from pax.utils import load
from pax.watchers import cg_visitation, ipd_visitation


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class EvalRunnerCG:
    """Holds the runner's state."""

    def __init__(self, args):
        self.args = args
        self.num_opps = args.num_opps
        self.eval_steps = 0
        self.eval_episodes = 0
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_datetime = datetime.now()
        self.start_time = time.time()
        self.train_steps = 0
        self.train_episodes = 0
        self.num_seeds = args.num_seeds
        self.run_path = args.run_path
        self.model_path = args.model_path
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)

    def eval_loop(self, env, agents, num_seeds, watchers):
        """Run evaluation of agents in environment"""

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
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=env.inner_episode_length,
            )

            # update second agent
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals

            # do second agent update
            final_t2 = t2._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], final_t2, a2_state, a2_mem
            )

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
            ), (*trajectories, a2_metrics)

        print("Evaluation")
        print("------------------------------")
        print(f"Number of Seeds: {self.num_seeds}")
        print(f"Number of Environments: {env.num_envs}")
        print(f"Number of Opponent: {self.num_opps}")
        print("-----------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        # Load model
        print("Local model_path", os.path.join(os.getcwd(), self.model_path))
        if watchers:
            wandb.restore(
                name=self.model_path, run_path=self.run_path, root=os.getcwd()
            )
        params = load(self.model_path)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        a1_state = a1_state._replace(params=params)

        mean_rewards_p1 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_rewards_p2 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_coop_prob_p1 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_coop_prob_p2 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_coins_per_episode_p1 = jnp.zeros(
            shape=(num_seeds, env.num_trials)
        )
        mean_coins_per_episode_p2 = jnp.zeros(
            shape=(num_seeds, env.num_trials)
        )
        for i in range(self.num_seeds):
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

            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals
            traj_1, traj_2, a2_metrics = stack

            if self.args.env_id == "coin_game":
                env_stats = self.cg_stats(env_state)
                rewards_0 = traj_1.rewards.sum(axis=1).mean()
                rewards_1 = traj_2.rewards.sum(axis=1).mean()

                mean_rewards_p1 = mean_rewards_p1

                mean_coop_prob_p1 = mean_coop_prob_p1.at[i, :].set(
                    env_stats["prob_coop/1"]
                )
                mean_coop_prob_p2 = mean_coop_prob_p2.at[i, :].set(
                    env_stats["prob_coop/2"]
                )
                mean_coins_per_episode_p1 = mean_coins_per_episode_p1.at[
                    i, :
                ].set(env_stats["coins_per_episode/1"])
                mean_coins_per_episode_p2 = mean_coins_per_episode_p2.at[
                    i, :
                ].set(env_stats["coins_per_episode/2"])

            elif self.args.env_type in [
                "meta",
                "sequential",
            ]:
                final_t1 = t1._replace(
                    step_type=2 * jnp.ones_like(t1.step_type)
                )

                env_stats = jax.tree_util.tree_map(
                    lambda x: x.item(), self.ipd_stats(traj_1, final_t1)
                )
                rewards_0 = traj_1.rewards.mean()
                rewards_1 = traj_2.rewards.mean()
            else:
                env_stats = {}

            print(f"Iteration: {i}")
            print(
                "--------------------------------------------------------------------------"
            )
            print(f"Episode Reward: {float(rewards_0), float(rewards_1)}")
            print(
                f"Cooperation Probability: {mean_coop_prob_p1[i].mean()}, {mean_coop_prob_p2[i].mean()}"
            )
            mean_coin_per_ep_p1 = mean_coins_per_episode_p1[i].mean()
            mean_coin_per_ep_p2 = mean_coins_per_episode_p2[i].mean()
            print(
                f"Coins per Episode: {mean_coin_per_ep_p1}, {mean_coin_per_ep_p2}"
            )
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                "--------------------------------------------------------------------------"
            )

            # trial rewards
            for out_step in range(env.num_trials):
                rewards_trial_mean_p1 = (
                    traj_1.rewards[out_step].sum(axis=0).mean()
                )
                rewards_trial_mean_p2 = (
                    traj_2.rewards[out_step].sum(axis=0).mean()
                )
                mean_rewards_p1 = mean_rewards_p1.at[i, out_step].set(
                    rewards_trial_mean_p1
                )
                mean_rewards_p2 = mean_rewards_p2.at[i, out_step].set(
                    rewards_trial_mean_p2
                )
                if out_step % 100 == 0:
                    print(f"Trial {out_step}")
                    print(
                        f"Reward | P1:{rewards_trial_mean_p1}, P2:{rewards_trial_mean_p2}"
                    )
                    coop_prob_p1 = mean_coop_prob_p1[i, out_step]
                    coop_prob_p2 = mean_coop_prob_p2[i, out_step]
                    print(
                        f"Probability of Cooperation | P1:{coop_prob_p1}, P2:{coop_prob_p2}"
                    )
                    coin_per_ep_p1 = mean_coins_per_episode_p1[i, out_step]
                    coin_per_ep_p2 = mean_coins_per_episode_p2[i, out_step]
                    print(
                        f"Coins per Episode | P1:{coin_per_ep_p1}, P2:{coin_per_ep_p2}"
                    )
                    print()

                if watchers:
                    eval_trial_log = {
                        "eval/trial": out_step + 1,
                        f"eval/reward_trial_p1_opp_{i}": rewards_trial_mean_p1,
                        f"eval/reward_trial_p2_opp_{i}": rewards_trial_mean_p2,
                        f"eval/reward_trial_p1_prob_{i}": coop_prob_p1,
                        f"eval/reward_trial_p2_prob_{i}": coop_prob_p2,
                        f"eval/coin_per_ep_p1_{i}": coin_per_ep_p1,
                        f"eval/coin_per_ep_p2_{i}": coin_per_ep_p2,
                    }
                    wandb.log(eval_trial_log)
                # TODO: Add step rewards?

            if watchers:
                wandb_log = {
                    "eval/time/minutes": float(
                        (time.time() - self.start_time) / 60
                    ),
                    "eval/time/seconds": float(
                        (time.time() - self.start_time)
                    ),
                    "eval/episode_reward/player_1": rewards_0,
                    "eval/episode_reward/player_2": rewards_1,
                }
                wandb_log = wandb_log | env_stats

                # player 2 metrics
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
                wandb.log(wandb_log)
            print()

        for out_step in range(env.num_trials):
            if watchers:
                wandb.log(
                    {
                        "eval/trial": out_step + 1,
                        "eval/reward/p1": mean_rewards_p1[:, out_step].mean(),
                        "eval/reward/p2": mean_rewards_p2[:, out_step].mean(),
                        "eval/prob_cooperation/p1": mean_coop_prob_p1[
                            :, out_step
                        ].mean(),
                        "eval/prob_cooperation/p2": mean_coop_prob_p2[
                            :, out_step
                        ].mean(),
                        "eval/coins_per_episode/p1": mean_coins_per_episode_p1[
                            :, out_step
                        ].mean(),
                        "eval/coins_per_episode/p2": mean_coins_per_episode_p2[
                            :, out_step
                        ].mean(),
                    }
                )

        return agents
