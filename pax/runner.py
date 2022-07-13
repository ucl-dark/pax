import time

from pax.env import IteratedPrisonersDilemma
from pax.independent_learners import IndependentLearners
from pax.strategies import TitForTat, Defect

import jax.numpy as jnp
import wandb

# TODO: make these a copy of acme


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
        for _ in range(int(num_episodes // env.num_envs)):
            rewards_0, rewards_1 = [], []
            t = env.reset()
            agents.log(watchers)
            while not (t[0].last()):
                actions = agents.select_action(t)
                t_prime = env.step(actions)
                r_0, r_1 = t_prime[0].reward, t_prime[1].reward

                # append step rewards to episode rewards
                rewards_0.append(r_0)
                rewards_1.append(r_1)

                # train model
                # TODO: Will this need to be changed to allow
                # for centralized training?
                agents.update(t, actions, t_prime)
                self.train_steps += 1

                # book keeping
                t = t_prime

                # logging
                if watchers:
                    agents.log(watchers)
                    wandb.log(
                        {
                            "train/training_steps": self.train_steps,
                            "train/step_reward/player_1": float(
                                jnp.array(r_0).mean()
                            ),
                            "train/step_reward/player_2": float(
                                jnp.array(r_1).mean()
                            ),
                            "time_elapsed_minutes": (
                                time.time() - self.start_time
                            )
                            / 60,
                            "time_elapsed_seconds": time.time()
                            - self.start_time,
                        }
                    )

            # end of episode stats
            self.train_episodes += 1
            rewards_0 = jnp.array(rewards_0)
            rewards_1 = jnp.array(rewards_1)

            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )
            if watchers:
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
                )
        print()
        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        for _ in range(num_episodes // env.num_envs):
            rewards_0, rewards_1 = [], []
            timesteps = env.reset()
            while not timesteps[0].last():
                actions = agents.select_action(timesteps)
                timesteps = env.step(actions)
                r_0, r_1 = timesteps[0].reward, timesteps[1].reward
                rewards_0.append(timesteps[0].reward)
                rewards_1.append(timesteps[1].reward)

                # update evaluation steps
                self.eval_steps += 1

                if watchers:
                    # agents.log(watchers)
                    wandb.log(
                        {
                            "eval/evaluation_steps": self.eval_steps,
                            "eval/step_reward/player_1": float(
                                jnp.array(r_0).mean()
                            ),
                            "eval/step_reward/player_2": float(
                                jnp.array(r_1).mean()
                            ),
                        }
                    )

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
