from re import T
import time

import jax

from pax.env import IteratedPrisonersDilemma
from pax.independent_learners import IndependentLearners
from pax.strategies import TitForTat, Defect

import jax.numpy as jnp
import wandb
from typing import NamedTuple
from dm_env import transition


# TODO: make these a copy of acme


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class Runner:
    """Holds the runner's state."""

    def __init__(self):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training ")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(1)

        def _inner_rollout(carry, unused):
            t1, t2, a1_state, a2_state, env_state = carry
            a1, new_a1_state, a1_extras = agent1._policy(
                a1_state.params, t1.observation, a1_state
            )
            a2, new_a2_state, a2_extras = agent2._policy(
                a2_state.params, t2.observation, a2_state
            )

            (tprime_1, tprime_2), env_state = env.runner_step(
                [
                    a1,
                    a2,
                ],
                env_state,
            )
            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                a1_extras["log_probs"],
                a1_extras["values"],
                tprime_1.last(),
                a1_state.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                a2_extras["log_probs"],
                a2_extras["values"],
                tprime_2.last(),
                None,
            )
            return (
                tprime_1,
                tprime_2,
                new_a1_state,
                new_a2_state,
                env_state,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            t1, t2, a1_state, a2_state, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                (t1, t2, a1_state, a2_state, env_state),
                None,
                length=env.inner_episode_length,
            )

            # do second agent update
            final_t2 = vals[1]._replace(step_type=2)
            a2_state = vals[3]
            a2_state = agent2.update(trajectories[1], final_t2, a2_state)
            return (t1, t2, a1_state, a2_state, env_state), trajectories

        @jax.jit
        def _meta_trajectory_reshape(batch_traj: Sample) -> Sample:
            batch_size = (
                batch_traj.observations.shape[0]
                * batch_traj.observations.shape[1]
            )
            return jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch_traj
            )

        for _ in range(0, max(int(num_episodes / env.num_envs), 1)):
            t_init = env.reset()
            env_state = env.state

            # uniquely nl code required here.
            a1_state = agent1._state
            a2_state = agent2.make_initial_state(
                rng, (env.observation_spec().num_values,)
            )
            rng, _ = jax.random.split(rng)

            # init hidden state to be empty
            # rollout episode
            xs = [None] * int(env.num_trials)
            ys = []
            carry = (*t_init, a1_state, a2_state, env_state)
            for x in xs:
                carry, y0 = _outer_rollout(carry, x)
                ys.append(y0)
            vals = carry
            obs, a, r, log_p, v, dones, h = [], [], [], [], [], [], []
            obs1, a1, r1, log_p1, v1, dones1, h1 = [], [], [], [], [], [], []

            for (y0, y1) in ys:
                obs.append(y0.observations)
                a.append(y0.actions)
                r.append(y0.rewards)
                log_p.append(y0.behavior_log_probs)
                v.append(y0.behavior_values)
                dones.append(y0.dones)
                h.append(y0.hiddens)

                obs1.append(y1.observations)
                a1.append(y1.actions)
                r1.append(y1.rewards)
                log_p1.append(y1.behavior_log_probs)
                v1.append(y1.behavior_values)
                dones1.append(y1.dones)
                h1.append(y1.hiddens)

            trajectories = (
                Sample(
                    jnp.stack(obs),
                    jnp.stack(a),
                    jnp.stack(r),
                    jnp.stack(log_p),
                    jnp.stack(v),
                    jnp.stack(dones),
                    jnp.stack(h),
                ),
                Sample(obs1, a1, jnp.stack(r1), log_p1, v1, dones1, h1),
            )

            # vals, trajectories = jax.lax.scan(
            #     _outer_rollout,
            #     (*t_init, a1_state, a2_state, env_state),
            #     None,
            #     length=env.num_trials,
            # )

            self.train_episodes += 1
            rewards_0 = trajectories[0].rewards.mean()
            rewards_1 = trajectories[1].rewards.mean()

            # update agent / add final trajectory
            final_t1 = vals[0]._replace(step_type=2)
            a1_state = vals[2]
            a1_state = agent1.update(
                _meta_trajectory_reshape(trajectories[0]), final_t1, a1_state
            )

            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )

            if watchers:
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
        agents.agents[1]._state = vals[3]

        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        for _ in range(max(num_episodes // env.num_envs, 1)):
            rewards_0, rewards_1 = [], []
            timesteps = env.reset()
            for _ in range(env.episode_length):
                actions = agents.select_action(timesteps)
                timesteps = env.step(actions)
                r_0, r_1 = timesteps[0].reward, timesteps[1].reward
                rewards_0.append(timesteps[0].reward)
                rewards_1.append(timesteps[1].reward)

                # update evaluation steps
                self.eval_steps += 1

                if watchers:
                    agents.log(watchers)
                    env_info = {
                        "eval/evaluation_steps": self.eval_steps,
                        "eval/step_reward/player_1": float(
                            jnp.array(r_0).mean()
                        ),
                        "eval/step_reward/player_2": float(
                            jnp.array(r_1).mean()
                        ),
                    }
                    wandb.log(env_info)

            # end of episode stats
            self.eval_episodes += 1
            rewards_0 = jnp.array(rewards_0)
            rewards_1 = jnp.array(rewards_1)

            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )
            if watchers:
                wandb.log(
                    {
                        "train/episodes": self.train_episodes,
                        "eval/episodes": self.eval_episodes,
                        "eval/episode_reward/player_1": float(
                            rewards_0.mean()
                        ),
                        "eval/episode_reward/player_2": float(
                            rewards_1.mean()
                        ),
                    }
                )
        agents.eval(False)
        print()
        return agents


if __name__ == "__main__":
    agents = IndependentLearners([Defect(), TitForTat()])
    # TODO: accept the arguments from config file instead of hard-coding
    # Default to prisoner's dilemma
    env = IteratedPrisonersDilemma(50, 5)
    wandb.init(mode="online")

    def log_print(agent) -> dict:
        print(agent)
        return None
