from datetime import datetime
import os
import time
from typing import List, NamedTuple

from dm_env import TimeStep
import jax
import jax.numpy as jnp
import wandb

from pax.utils import load

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


class EvalRunner:
    """Holds the runner's state."""

    def __init__(self, args):
        self.algo = args.es.algo
        self.args = args
        self.num_opps = args.num_opps
        self.eval_steps = 0
        self.eval_episodes = 0
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_datetime = datetime.now()
        self.start_time = time.time()
        self.train_steps = 0
        self.train_episodes = 0
        self.num_iters = args.num_iters
        self.run_path = args.run_path
        self.filename = args.filename

        def _reshape_opp_dim(x):
            # x: [num_opps, num_envs ...]
            # x: [batch_size, ...]
            batch_size = args.num_envs * args.num_opps
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), x
            )

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

        self.state_visitation = _state_visitation
        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)

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
        num_iters = self.num_iters
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Number of Seeds: {num_iters}")
        print(f"Number of Meta Episodes: {num_iters}")
        print(f"Number of Environments: {env.num_envs}")
        print(f"Number of Opponent: {self.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        # Load agent
        raw_artifact = wandb.run.use_artifact(self.run_path)
        model_path = raw_artifact.get_path(self.filename)
        model_path.download(root=self.args.save_dir)

        # LOAD
        params = load(os.path.join(self.args.save_dir, self.filename))

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        a1_state = a1_state._replace(params=params)

        # Track mean rewards and state visitations
        mean_rewards_p1 = jnp.zeros(shape=(num_iters, env.num_trials))
        mean_rewards_p2 = jnp.zeros(shape=(num_iters, env.num_trials))
        mean_visits = jnp.zeros(shape=(num_iters, env.num_trials, 5))
        mean_state_freq = jnp.zeros(shape=(num_iters, env.num_trials, 5))
        mean_cooperation_prob = jnp.zeros(shape=(num_iters, env.num_trials, 5))

        for opp_i in range(num_iters):
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
            a1_mem = agent1.batch_reset(a1_mem, True)

            # update second agent
            a2_state, a2_mem = vals[4], vals[5]

            # logging
            self.train_episodes += 1
            rewards_0 = stack[0].rewards.mean()
            rewards_1 = stack[1].rewards.mean()

            if opp_i % log_interval == 0:
                # Logging step rewards loop
                # Shape: [outer_loop, inner_loop, popsize, num_opps, num_envs, ...]
                visits = self.state_visitation(
                    traj_1.observations, traj_1.actions, final_t1.observation
                )
                states = visits.reshape((int(visits.shape[0] / 2), 2)).sum(
                    axis=1
                )
                state_freq = states / states.sum()
                action_probs = visits[::2].astype("float32") / states
                action_probs = jnp.nan_to_num(action_probs).astype("int32")
                print(f"Summary | Opponent: {opp_i+1}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"{self.args.agent1}: {rewards_0} | {self.args.agent2}: {rewards_1}"
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
                action_trial_probs = (
                    visits_trial[::2].astype("float32") / states_trial
                )
                action_trial_probs = jnp.nan_to_num(action_trial_probs).astype(
                    "int32"
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
