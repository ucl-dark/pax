import jax.numpy as jnp

import wandb

from src.game import IteratedPrisonersDilemma
from src.strategies import Altruistic, TitForTat, Defect


def train(env, agent_0, agent_1, num_episodes):
    print(f"Training ")
    print("-----------------------")

    for _ in range(num_episodes // (env.num_envs + 1)):
        rewards_0, rewards_1 = [], []
        t1, t2 = env.reset()

        while not (t1.last() or t2.last()):
            action_0, _ = agent_0.select_action(t1)
            action_1, _ = agent_1.select_action(t2)
            t1, t2 = env.step(action_0, action_1)

            rewards_0.append(t1.reward)
            rewards_1.append(t2.reward)

            # book keeping
            obs_0, obs_1 = t1.observation, t2.observation

            # train model
            agent_0.update()
            agent_1.update()

            # step logging
            wandb.log(policy_logger(agent_0))
            wandb.log(value_logger(agent_0))
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

        wandb.log(
            {
                "Episode Reward Player 1": float(rewards_0.mean()),
                "Episode Reward Player 2": float(rewards_1.mean()),
            }
        )
    return agent_0, agent_1


def eval(env, agent_0, agent_1, num_episodes, logging):
    print(f"Evaluating")
    print("-----------------------")

    for _ in range(num_episodes // (env.num_envs + 1)):
        rewards_0, rewards_1 = [], []
        t1, t2 = env.reset()

        while not (t1.last() or t2.last()):
            a1 = agent_0.select_action(t1)
            a2 = agent_1.select_action(t2)
            t1, t2 = env.step(a1, a2)

            rewards_0.append(t1.reward)
            rewards_1.append(t2.reward)

            # step logging
            if logging:
                wandb.log(policy_logger(agent_0))
                wandb.log(value_logger(agent_0))
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
        if logging:
            wandb.log(
                {
                    "Episode Reward Player 1": float(rewards_0.mean()),
                    "Episode Reward Player 2": float(rewards_1.mean()),
                }
            )
    return agent_0, agent_1


if __name__ == "__main__":
    agent_0 = Altruistic()
    agent_1 = Defect()
    env = IteratedPrisonersDilemma(50, 5)
    eval(env, agent_0, agent_1, 50, False)
