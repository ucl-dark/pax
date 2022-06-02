import jax.numpy as jnp
import wandb
import jax.random
import jax

from .env import IteratedPrisonersDilemma
from .strategies import Altruistic, TitForTat, Defect
from .independent_learners import IndependentLearners

# TODO: make these a copy of acme


class Runner:
    """Holds the runner's state."""

    def __init__(self):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0

    def train_loop(self, env, agents, num_episodes, watchers, key):
        """Run training of agents in environment"""
        print("Training ")
        print("-----------------------")
        for _ in range(int(num_episodes // env.num_envs)):
            rewards_0, rewards_1 = [], []
            t = env.reset()
            while not (t[0].last()):
                key, key2 = jax.random.split(key)
                actions = agents.select_action(key, t)
                t_prime = env.step(actions)
                r_0, r_1 = t_prime[0].reward, t_prime[1].reward

                # append step rewards to episode rewards
                rewards_0.append(r_0)
                rewards_1.append(r_1)

                # train model
                agents.update(t, actions, t_prime, key2)
                self.train_steps += 1

                # book keeping
                t = t_prime

                # logging
                if watchers:
                    # plot against number of training steps
                    agents.log(watchers)
                    wandb.log(
                        {
                            "train/training_steps": self.train_steps,
                            "train/step_reward/player_1": float(  # this is the avg. reward across envs
                                jnp.array(r_0).mean()
                            ),
                            "train/step_reward/player_2": float(
                                jnp.array(r_1).mean()
                            ),
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
        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers, key):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        for _ in range(num_episodes // env.num_envs):
            rewards_0, rewards_1 = [], []
            timesteps = env.reset()

            while not timesteps[0].last():
                key, key2 = jax.random.split(key)
                actions = agents.select_action(key, timesteps)
                timesteps = env.step(actions)

                r_0, r_1 = timesteps[0].reward, timesteps[1].reward

                rewards_0.append(timesteps[0].reward)
                rewards_1.append(timesteps[1].reward)

                # update evaluation steps
                self.eval_steps += 1

                if watchers:
                    agents.log(watchers)
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
        return agents


if __name__ == "__main__":
    agents = IndependentLearners([Defect(), TitForTat()])
    # TODO: accept the arguments from config file instead of hard-coding
    # 50 steps per episode for 10000 episodes makes for 50k steps
    env = IteratedPrisonersDilemma(50, 5)
    wandb.init(mode="online")

    def log_print(agent) -> dict:
        print(agent)
        return None
