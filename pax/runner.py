import jax.numpy as jnp
import wandb

from .env import IteratedPrisonersDilemma
from .strategies import Altruistic, TitForTat, Defect
from .independent_learners import IndependentLearners

# TODO: make these a copy of acme

def train_loop(env, agents, num_episodes, watchers):
    """Run training of agents in environment"""
    print("Training ")
    print("-----------------------")
    for _ in range(int(num_episodes // env.num_envs)):
        rewards_0, rewards_1 = [], []
        t = env.reset()
        while not (t[0].last()):
            actions = agents.select_action(t)

            t_prime = env.step(actions)

            r_0, r_1 = t_prime[0].reward, t_prime[1].reward
            rewards_0.append(r_0)
            rewards_1.append(r_1)

            # train model
            agents.update(t, actions, t_prime)

            # book keeping
            t = t_prime

            # logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "train/step_reward/player_1": float(
                            jnp.array(r_0).mean()
                        ),
                        "train/step_reward/player_2": float(
                            jnp.array(r_1).mean()
                        ),
                    }
                )

        # end of episode stats
        rewards_0 = jnp.array(rewards_0)
        rewards_1 = jnp.array(rewards_1)

        print(
            f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
        )
        if watchers:
            wandb.log(
                {
                    "train/episode_reward/player_1": float(rewards_0.mean()),
                    "train/episode_reward/player_2": float(rewards_1.mean()),
                }
            )
    return agents


def evaluate_loop(env, agents, num_episodes, watchers):
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

            rewards_0.append(timesteps[0].reward)
            rewards_1.append(timesteps[1].reward)

            # step logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "eval/step_reward/player_1": float(
                            jnp.array(rewards_0).mean()
                        ),
                        "eval/step_reward/player_2": float(
                            jnp.array(rewards_1).mean()
                        ),
                    }
                )
        # end of episode stats
        rewards_0 = jnp.array(rewards_0)
        rewards_1 = jnp.array(rewards_1)

        print(
            f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
        )
        if watchers:
            wandb.log(
                {
                    "eval/episode_reward/player_1": float(rewards_0.mean()),
                    "eval/episode_reward/player_2": float(rewards_1.mean()),
                }
            )
    agents.eval(False)
    return agents


if __name__ == "__main__":
    agents = IndependentLearners([Defect(), TitForTat()])
    env = IteratedPrisonersDilemma(50, 5)
    #wandb.init(mode="offline")
    wandb.init(mode="online")

    def log_print(agent) -> dict:
        print(agent)
        return None

    evaluate_loop(env, agents, 50, None)
    train_loop(env, agents, 2, [log_print, log_print])
