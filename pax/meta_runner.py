import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

import wandb


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
def _meta_trajectory_reshape(batch_traj: Sample) -> Sample:
    batch_size = (
        batch_traj.observations.shape[0] * batch_traj.observations.shape[1]
    )
    return jax.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch_traj
    )


class MetaRunner:
    """Holds the runner's state."""

    def __init__(self, args):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()
        self.args = args

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(0)

        def _inner_rollout(carry, unused):
            t1, t2, a1_state, a2_state, env_state = carry
            a1, new_a1_state = agent1._policy(
                a1_state.params, t1.observation, a1_state
            )
            a2, new_a2_state = agent2._policy(
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
                new_a1_state.extras["log_probs"],
                new_a1_state.extras["values"],
                tprime_1.last() * jnp.zeros(env.num_envs),
                a1_state.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                new_a2_state.extras["log_probs"],
                new_a2_state.extras["values"],
                tprime_2.last(),
                a2_state.hidden,
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

        # run actual loop
        for _ in range(0, max(int(num_episodes / env.num_envs), 1)):
            t_init = env.reset()
            env_state = env.state

            # uniquely nl code required here.
            a1_state = agent1.reset_memory()
            if self.args.agent2 == "Naive":
                a2_state = agent2.make_initial_state(
                    rng, (env.observation_spec().num_values,)
                )
            else:
                a2_state = agent2.reset_memory()
            rng, _ = jax.random.split(rng)

            # rollout outer episode
            vals, trajectories = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a2_state, env_state),
                None,
                length=env.num_trials,
            )
            self.train_episodes += 1
            rewards_0 = trajectories[0].rewards.mean()
            rewards_1 = trajectories[1].rewards.mean()
            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )

            # update outer agent
            final_t1 = vals[0]._replace(step_type=2)
            a1_state = vals[2]
            a1_state = agent1.update(
                _meta_trajectory_reshape(trajectories[0]), final_t1, a1_state
            )

            # logging
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
        # update agents
        agents.agents[0]._state = a1_state
        agents.agents[1]._state = a2_state
        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        agent1, agent2 = agents.agents
        agent1.reset_memory()
        agent2.reset_memory()

        for _ in range(num_episodes // env.num_envs):
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
                        "eval/joint_reward": float(
                            rewards_0.mean() + rewards_1.mean() * 0.5
                        ),
                    }
                )
        agents.eval(False)
        print()
        return agents
