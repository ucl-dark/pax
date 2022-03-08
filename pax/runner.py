import jax.numpy as jnp

import wandb
from .watchers import policy_logger, value_logger

from .game import IteratedPrisonersDilemma
from .strategies import Altruistic, TitForTat, Defect
from .independent_learners import IndependentLearners


def train_loop(env, agents, num_episodes, watchers):
    """Run training of agents in environment
    TODO: make this a copy of acme
    """
    print("Training ")
    print("-----------------------")

    for _ in range((num_episodes // env.num_envs) + 1):
        rewards_0, rewards_1 = [], []
        t = env.reset()
        while not (t[0].last()):
            actions = agents.select_action(t)
            t_prime = env.step(actions)

            r_0, r_1 = t_prime[0].reward, t_prime[1].reward
            rewards_0.append(r_0)
            rewards_1.append(r_1)

            # train model
            agents.update(t, t_prime)

            # book keeping
            t = t_prime

            # logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "Reward Player 1": float(jnp.array(r_0).mean()),
                        "Reward Player 2": float(jnp.array(r_1).mean()),
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
                    "Episode Reward Player 1": float(rewards_0.mean()),
                    "Episode Reward Player 2": float(rewards_1.mean()),
                }
            )
    return agents


def evaluate_loop(env, agents, num_episodes, watchers):
    """Run evaluation of agents against environment"""
    print("Evaluating")
    print("-----------------------")
    for _ in range(num_episodes // (env.num_envs + 1)):
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
                        "Reward Player 1": float(jnp.array(rewards_0).mean()),
                        "Reward Player 2": float(jnp.array(rewards_1).mean()),
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
                    "Episode Reward Player 1": float(rewards_0.mean()),
                    "Episode Reward Player 2": float(rewards_1.mean()),
                }
            )
    return agents


if __name__ == "__main__":
    agents = IndependentLearners([Altruistic(), TitForTat()])
    env = IteratedPrisonersDilemma(50, 5)
    wandb.init(mode="offline")

    def log_sac(agent) -> dict:
        policy_dict = policy_logger(agent)
        value_dict = value_logger(agent)
        logger_dict = policy_dict.update(value_dict)
        return logger_dict

    def log_print(agent) -> dict:
        print(agent)
        return None

    evaluate_loop(env, agents, 50, None)
    train_loop(env, agents, 2, [log_print, log_print])
