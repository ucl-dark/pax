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

        def _state_visitation(traj: Sample, final_t: TimeStep) -> List:
            # obs [num_outer_steps, num_inner_steps, num_opps, num_envs, ...]
            # final_t [num_opps, num_envs, ...]
            num_timesteps = (
                traj.observations.shape[0] * traj.observations.shape[1]
            )
            obs = jnp.reshape(
                traj.observations,
                (num_timesteps,) + traj.observations.shape[2:],
            )
            final_obs = jax.lax.expand_dims(final_t.observation, [0])
            obs = jnp.argmax(jnp.append(obs, final_obs, axis=0), axis=-1)
            return jnp.bincount(obs.flatten(), length=5)

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
            a2_state, a2_mem = agent2.batch_update(
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
            ), trajectories

        print("Training")
        print("-----------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        # Initialize eval for evolutionary algo
        num_gens = 20
        popsize = 1
        num_opps = 1
        param_reshaper = ParameterReshaper(agent1._state.params)
        es_logging = ESLog(
            param_reshaper.total_params,
            num_gens,
            top_k=self.top_k,
            maximize=True,
        )

        # Evolution specific: add pop size dimension
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step),
        )

        # Load agent
        # TODO: Temporary directory before figuring out saving.
        eval_dir = (
            "/Users/newtonkwan/UCL/Research/pax/exp/EARL-PPO_memory-vs-Naive/"
        )
        "run-seed-0-OpenES-pop-size-10"
        "-num-opps-1/2022-08-19_01:35:55.912020/generation_0"
        log = es_logging.load(os.path.join(eval_dir, self.args.model_path))

        # TODO: Why can't this be moved to EvolutionaryLearners?
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

        for opp_i in range(20):
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

            vals, trajectories = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            # a1, a1_state, a1_new_mem = agent1.batch_policy(
            #     a1_state,
            #     t_init[0].observation,
            #     a1_mem,
            # )

            # Fitness
            fitness = trajectories[0].rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = trajectories[1].rewards.mean(axis=(0, 1, 3, 4))

            # Calculate state visitations
            final_t1 = vals[0]._replace(
                step_type=2 * jnp.ones_like(vals[0].step_type)
            )
            visits = self.state_visitation(trajectories[0], final_t1)
            prob_visits = visits / visits.sum()

            print(f"Summary | Number of opponents: {num_opps}")
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"EARL: {fitness.mean()} | Naive Learners: {other_fitness.mean()}"
            )
            # print(f"State Frequency: {visits}")
            print(f"State Visitation: {prob_visits}")
            print(
                "--------------------------------------------------------------------------"
            )

            # Logging step rewards loop
            # Shape: [outer_loop, inner_loop, popsize, num_opps, num_envs]
            observations = self.get_observations(trajectories[0])
            p1_rewards = trajectories[0].rewards
            p2_rewards = trajectories[1].rewards

            obs_opp_i = observations[:, :, :, opp_i, :].squeeze()
            p1_rew_opp_i = p1_rewards[:, :, :, opp_i, :].squeeze()
            p2_rew_opp_i = p2_rewards[:, :, :, opp_i, :].squeeze()
            inner_steps = 0
            for out_step in range(env.num_trials):
                obs_outer_opp_i = obs_opp_i[out_step]
                p1_ep_rew_opp_i = p1_rew_opp_i[out_step]
                p2_ep_rew_opp_i = p2_rew_opp_i[out_step]
                p1_ep_mean_rew_opp_i = p1_rew_opp_i[out_step].mean()
                p2_ep_mean_rew_opp_i = p2_rew_opp_i[out_step].mean()
                state_visit_trial_i = self.get_state_visitation(
                    obs_outer_opp_i
                )

                # Over trials
                state_visit_avg = (
                    self.get_state_visitation(observations[out_step])
                    / num_opps
                )
                state_visit_prob = state_visit_avg / state_visit_avg.sum()
                p1_ep_rew_mean = p1_rewards[out_step].mean()
                p2_ep_rew_mean = p2_rewards[out_step].mean()
                p1_ep_rew_median = jnp.median(
                    p1_rewards[out_step].mean(axis=(0, 1, 3))
                )
                p2_ep_rew_median = jnp.median(
                    p2_rewards[out_step].mean(axis=(0, 1, 3))
                )
                prob_visits = state_visit_trial_i / state_visit_trial_i.sum()
                if watchers:
                    wandb.log(
                        {
                            "eval/trial": out_step + 1,
                            "eval/reward/player_1_mean": p1_ep_rew_mean,
                            "eval/reward/player_2_mean": p2_ep_rew_mean,
                            "eval/reward/player_1_median": p1_ep_rew_median,
                            "eval/reward/player_2_median": p2_ep_rew_median,
                            f"eval/reward_trial/player_1_opp_{opp_i+1}": p1_ep_mean_rew_opp_i,
                            f"eval/reward_trial/player_2_opp_{opp_i+1}": p2_ep_mean_rew_opp_i,
                            "eval/state_visitation/CC": state_visit_avg[0],
                            "eval/state_visitation/CD": state_visit_avg[1],
                            "eval/state_visitation/DC": state_visit_avg[2],
                            "eval/state_visitation/DD": state_visit_avg[3],
                            "eval/state_visitation/START": state_visit_avg[4],
                            "eval/state_visitation/CC_prob": state_visit_prob[
                                0
                            ],
                            "eval/state_visitation/CD_prob": state_visit_prob[
                                1
                            ],
                            "eval/state_visitation/DC_prob": state_visit_prob[
                                2
                            ],
                            "eval/state_visitation/DD_prob": state_visit_prob[
                                3
                            ],
                            "eval/state_visitation/START_prob": state_visit_prob[
                                4
                            ],
                            f"eval/state_visitation_trial/CC_opp_{opp_i+1}": state_visit_trial_i[
                                0
                            ],
                            f"eval/state_visitation_trial/CD_opp_{opp_i+1}": state_visit_trial_i[
                                1
                            ],
                            f"eval/state_visitation_trial/DC_opp_{opp_i+1}": state_visit_trial_i[
                                2
                            ],
                            f"eval/state_visitation_trial/DD_opp_{opp_i+1}": state_visit_trial_i[
                                3
                            ],
                            f"eval/state_visitation_trial/START_opp{opp_i+1}": state_visit_trial_i[
                                4
                            ],
                            f"eval/state_visitation_trial/CC_probability_opp_{opp_i+1}": prob_visits[
                                0
                            ],
                            f"eval/state_visitation_trial/CD_probability_opp_{opp_i+1}": prob_visits[
                                1
                            ],
                            f"eval/state_visitation_trial/DC_probability_opp_{opp_i+1}": prob_visits[
                                2
                            ],
                            f"eval/state_visitation_trial/DD_probability_opp_{opp_i+1}": prob_visits[
                                3
                            ],
                            f"eval/state_visitation_trial/START_probability_opp_{opp_i+1}": prob_visits[
                                4
                            ],
                        }
                    )

                for in_step in range(env.inner_episode_length):
                    p1_step_rew_opp_i = p1_ep_rew_opp_i[in_step]
                    p2_step_rew_opp_i = p2_ep_rew_opp_i[in_step]
                    if watchers:
                        wandb.log(
                            {
                                "eval/timestep": inner_steps + 1,
                                f"eval/reward_step/player_1_opp_{opp_i+1}": p1_step_rew_opp_i,
                                f"eval/reward_step/player_2_opp_{opp_i+1}": p2_step_rew_opp_i,
                            }
                        )
                    inner_steps += 1

            state_visit_opp_i = self.get_state_visitation(obs_opp_i)
            prob_visits = state_visit_opp_i / state_visit_opp_i.sum()
            print(
                f"Opponent: {opp_i+1} | EARL: {p1_rew_opp_i.mean()} "
                f"| Naive Learner {opp_i+1}: {p2_rew_opp_i.mean()}"
            )
            print(f"State visitations: {prob_visits}")
            if watchers:
                wandb.log(
                    {
                        "eval/time/minutes": float(
                            (time.time() - self.start_time) / 60
                        ),
                        "eval/time/seconds": float(
                            (time.time() - self.start_time)
                        ),
                    }
                )

        if watchers:
            agents.log(watchers)
        print()
        return agents
