from datetime import datetime
import os
import time
from typing import List, NamedTuple

from dm_env import TimeStep
from evosax import FitnessShaper, ParameterReshaper
import jax
import jax.numpy as jnp
import wandb

# TODO: import when evosax library is updated
# from evosax.utils import ESLog
from pax.watchers import ESLog

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


class EvalRunner:
    """Holds the runner's state."""

    def __init__(self, args):
        self.algo = args.es.algo
        self.args = args
        self.generations = 0
        self.num_opps = args.num_opps
        self.eval_steps = 0
        self.eval_episodes = 0
        self.popsize = args.popsize
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_datetime = datetime.now()
        self.start_time = time.time()
        self.top_k = args.top_k
        self.train_steps = 0
        self.train_episodes = 0
        self.num_iters = args.num_iters
        self.run_path = args.run_path
        self.filename = args.filename

        def _state_visitation(
            observations: jnp.ndarray,
            actions: jnp.ndarray,
            final_obs: jnp.ndarray,
        ) -> List:
            # obs [num_outer_steps, num_inner_steps, num_opps, num_envs, ...]
            # final_t [num_opps, num_envs, ...]
            num_timesteps = observations.shape[0] * observations.shape[1]
            # obs = [0, 1, 2, 3, 4], a = [0, 1]
            # combine = [0, .... 9]
            state_actions = 2 * jnp.argmax(observations, axis=-1) + actions
            state_actions = jnp.reshape(
                state_actions,
                (num_timesteps,) + state_actions.shape[2:],
            )
            # assume final step taken is cooperate
            final_obs = jax.lax.expand_dims(
                2 * jnp.argmax(final_obs, axis=-1), [0]
            )
            state_actions = jnp.append(state_actions, final_obs, axis=0)
            return jnp.bincount(state_actions.flatten(), length=10)

        self.state_visitation = jax.jit(_state_visitation)

    def eval_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""

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
        print("-----------------------")
        # Initialize agents and RNG
        num_iters = num_gens = 20
        num_iters = num_gens = self.num_iters
        popsize = 1
        num_opps = 1
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Number of Seeds: {num_iters}")
        print(f"Number of Meta Episodes: {num_iters}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {env.num_envs}")
        print(f"Number of Opponent: {self.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        # Initialize eval for evolutionary algo
        param_reshaper = ParameterReshaper(agent1._state.params)
        es_logging = ESLog(
            param_reshaper.total_params,
            num_gens,
            top_k=self.top_k,
            maximize=True,
        )

        # Load agent

        model_path = wandb.restore(name=self.filename, run_path=self.run_path)
        # TODO: Full test
        print(f"Loading from run: {self.run_path}")
        print(f"Filepath: {model_path.name}")
        log = es_logging.load(model_path.name)

        # TODO: Remove this local test
        # testing_dir = f"/Users/newtonkwan/UCL/Research/pax/pax/test_model/generation_9300"
        # log = es_logging.load(testing_dir)

        # Evolution specific: add pop size dimension
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step),
        )

        # Evolution specific: Initialize batch over popsize
        init_hidden = jnp.tile(
            agent1._mem.hidden,
            (popsize, num_opps, 1, 1),
        )
        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(agent1._state.random_key, popsize),
            init_hidden,
        )

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        params = param_reshaper.reshape(log["top_gen_params"][0:1])
        a1_state = a1_state._replace(params=params)

        # Track mean rewards and state visitations
        mean_rewards_p1 = jnp.zeros(shape=(num_iters, env.num_trials))
        mean_rewards_p2 = jnp.zeros(shape=(num_iters, env.num_trials))
        mean_visits = jnp.zeros(shape=(num_iters, env.num_trials, 5))
        mean_state_freq = jnp.zeros(shape=(num_iters, env.num_trials, 5))
        mean_cooperation_prob = jnp.zeros(shape=(num_iters, env.num_trials, 5))

        for opp_i in range(num_iters):
            rng, rng_run, rng_key = jax.random.split(rng, 3)
            t_init, env_state = env.runner_reset(
                (popsize, num_opps, env.num_envs), rng_run
            )

            # Player 1
            a1_mem = agent1.batch_reset(a1_mem, True)

            # Player 2
            a2_state, a2_mem = agent2.batch_init(
                jax.random.split(rng_key, popsize * num_opps).reshape(
                    self.popsize, num_opps, -1
                ),
                a2_mem.hidden,
            )
            a2_mem = agent2.batch_reset(a2_mem, True)

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            traj_1, traj_2, a2_metrics = stack

            # Fitness
            fitness = traj_1.rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = traj_2.rewards.mean(axis=(0, 1, 3, 4))

            # Calculate state visitations
            final_t1 = vals[0]._replace(
                step_type=2 * jnp.ones_like(vals[0].step_type)
            )

            # Logging step rewards loop
            # Shape: [outer_loop, inner_loop, popsize, num_opps, num_envs, ...]
            visits = self.state_visitation(
                traj_1.observations, traj_1.actions, final_t1.observation
            )
            states = visits.reshape((int(visits.shape[0] / 2), 2)).sum(axis=1)
            state_freq = states / states.sum()
            action_probs = visits[::2] / jax.lax.select(
                states > 0, states, jnp.ones_like(states)
            )

            print(f"Summary | Opponent: {opp_i+1}")
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"EARL: {fitness.mean()} | Naive Learner: {other_fitness.mean()}"
            )
            print(f"State Visits: {states}")
            print(f"State Frequency: {state_freq}")
            print(f"Cooperation Probability: {action_probs}")
            print(
                "--------------------------------------------------------------------------"
            )

            inner_steps = 0
            for out_step in range(env.num_trials):
                rewards_trial_mean_p1 = traj_1.rewards[out_step].mean()
                rewards_trial_mean_p2 = traj_2.rewards[out_step].mean()
                visits_trial = self.state_visitation(
                    traj_1.observations[out_step],
                    traj_1.actions[out_step],
                    final_t1.observation[out_step],
                )
                states_trial = visits_trial.reshape(
                    (int(visits_trial.shape[0] / 2), 2)
                ).sum(axis=1)
                state_trial_freq = states_trial / states_trial.sum()
                action_trial_probs = visits_trial[::2] / jax.lax.select(
                    states_trial > 0, states_trial, jnp.ones_like(states_trial)
                )

                if watchers:
                    wandb.log(
                        {
                            "eval/trial": out_step + 1,
                            f"eval/reward_trial/player_1_opp_{opp_i+1}": rewards_trial_mean_p1,
                            f"eval/reward_trial/player_2_opp_{opp_i+1}": rewards_trial_mean_p2,
                            f"eval/state_visitation_trial/CC_opp_{opp_i+1}": states_trial[
                                0
                            ],
                            f"eval/state_visitation_trial/CD_opp_{opp_i+1}": states_trial[
                                1
                            ],
                            f"eval/state_visitation_trial/DC_opp_{opp_i+1}": states_trial[
                                2
                            ],
                            f"eval/state_visitation_trial/DD_opp_{opp_i+1}": states_trial[
                                3
                            ],
                            f"eval/state_visitation_trial/START_opp{opp_i+1}": states_trial[
                                4
                            ],
                            f"eval/state_visitation_trial/CC_freq_opp_{opp_i+1}": state_trial_freq[
                                0
                            ],
                            f"eval/state_visitation_trial/CD_freq_opp_{opp_i+1}": state_trial_freq[
                                1
                            ],
                            f"eval/state_visitation_trial/DC_freq_opp_{opp_i+1}": state_trial_freq[
                                2
                            ],
                            f"eval/state_visitation_trial/DD_freq_opp_{opp_i+1}": state_trial_freq[
                                3
                            ],
                            f"eval/state_visitation_trial/START_freq_opp_{opp_i+1}": state_trial_freq[
                                4
                            ],
                            "eval/cooperation_probability_trial/"
                            f"CC_coop_prob_opp_{opp_i+1}": action_trial_probs[
                                0
                            ],
                            "eval/cooperation_probability_trial/"
                            f"CD_coop_prob_opp_{opp_i+1}": action_trial_probs[
                                1
                            ],
                            "eval/cooperation_probability_trial/"
                            f"DC_coop_prob_opp_{opp_i+1}": action_trial_probs[
                                2
                            ],
                            "eval/cooperation_probability_trial/"
                            f"DD_coop_prob_opp_{opp_i+1}": action_trial_probs[
                                3
                            ],
                            "eval/cooperation_probability_trial/"
                            f"START_coop_prob_opp_{opp_i+1}": action_trial_probs[
                                4
                            ],
                        }
                    )

                for in_step in range(env.inner_episode_length):
                    rewards_step_p1 = traj_1.rewards[out_step, in_step]
                    rewards_step_p2 = traj_2.rewards[out_step, in_step]
                    if watchers:
                        wandb.log(
                            {
                                "eval/timestep": inner_steps + 1,
                                f"eval/reward_step/player_1_opp_{opp_i+1}": rewards_step_p1,
                                f"eval/reward_step/player_2_opp_{opp_i+1}": rewards_step_p2,
                            }
                        )
                    inner_steps += 1

                mean_rewards_p1 = mean_rewards_p1.at[opp_i, out_step].set(
                    rewards_trial_mean_p1
                )  # jnp.zeros(shape=(num_iters, env.num_trials))
                mean_rewards_p2 = mean_rewards_p2.at[opp_i, out_step].set(
                    rewards_trial_mean_p2
                )
                mean_visits = mean_visits.at[opp_i, out_step, :].set(
                    states_trial
                )  # jnp.zeros(shape=(num_iters, env.num_trials, 5))
                mean_state_freq = mean_state_freq.at[opp_i, out_step, :].set(
                    state_trial_freq
                )
                mean_cooperation_prob = mean_cooperation_prob.at[
                    opp_i, out_step, :
                ].set(action_trial_probs)

            if watchers:
                # metrics [outer_timesteps, num_opps]
                flattened_metrics = jax.tree_util.tree_map(
                    lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                )

                agent2._logger.metrics.update(flattened_metrics)
                agent1._logger.metrics.update(flattened_metrics)
                # TODO: Only works on 3.9 and i can't get it on colab
                # agent2._logger.metrics = (
                #     agent2._logger.metrics | flattened_metrics
                # )

                # agent1._logger.metrics = (
                #     agent1._logger.metrics | flattened_metrics
                # )
                agents.log(watchers)
                wandb.log(
                    {
                        "eval/opponent": opp_i + 1,
                        "eval/time/minutes": float(
                            (time.time() - self.start_time) / 60
                        ),
                        "eval/time/seconds": float(
                            (time.time() - self.start_time)
                        ),
                    }
                )
        for out_step in range(env.num_trials):
            if watchers:
                wandb.log(
                    {
                        "eval/trial": out_step + 1,
                        "eval/reward/p1": mean_rewards_p1[:, out_step].mean(),
                        "eval/reward/p2": mean_rewards_p2[:, out_step].mean(),
                        # TODO: Median?
                        "eval/state_visitation/CC": mean_visits[
                            :, out_step, 0
                        ].mean(),
                        "eval/state_visitation/CD": mean_visits[
                            :, out_step, 1
                        ].mean(),
                        "eval/state_visitation/DC": mean_visits[
                            :, out_step, 2
                        ].mean(),
                        "eval/state_visitation/DD": mean_visits[
                            :, out_step, 3
                        ].mean(),
                        "eval/state_visitation/START": mean_visits[
                            :, out_step, 4
                        ].mean(),
                        "eval/state_visitation/CC_freq": mean_state_freq[
                            :, out_step, 0
                        ].mean(),
                        "eval/state_visitation/CD_freq": mean_state_freq[
                            :, out_step, 1
                        ].mean(),
                        "eval/state_visitation/DC_freq": mean_state_freq[
                            :, out_step, 2
                        ].mean(),
                        "eval/state_visitation/DD_freq": mean_state_freq[
                            :, out_step, 3
                        ].mean(),
                        "eval/state_visitation/START_freq": mean_state_freq[
                            :, out_step, 4
                        ].mean(),
                        "eval/cooperation_probability/CC": mean_cooperation_prob[
                            :, out_step, 0
                        ].mean(),
                        "eval/cooperation_probability/CD": mean_cooperation_prob[
                            :, out_step, 1
                        ].mean(),
                        "eval/cooperation_probability/DC": mean_cooperation_prob[
                            :, out_step, 2
                        ].mean(),
                        "eval/cooperation_probability/DD": mean_cooperation_prob[
                            :, out_step, 3
                        ].mean(),
                        "eval/cooperation_probability/START": mean_cooperation_prob[
                            :, out_step, 4
                        ].mean(),
                    }
                )

        print()
        return agents
