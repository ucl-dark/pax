import jax.numpy as jnp

import wandb
from pax.experiments import policy_logger, value_logger

from pax.game import IteratedPrisonersDilemma
from pax.strategies import Altruistic, TitForTat, Defect
from pax.independent_learners import IndependentLeaners


def train_loop(env, agent_0, agent_1, num_episodes):
    """Run training of agents in environment
    TODO: make this a copy of acme
    """
    print(f"Training ")
    print("-----------------------")

    agents = IndependentLeaners([agent_0, agent_1])

    for _ in range(num_episodes // (env.num_envs + 1)):
        rewards_0, rewards_1 = [], []
        t1, t2 = env.reset()

        while not (t1.last() or t2.last()):
            action_0, _ = agent_0.select_action(t1)
            action_1, _ = agent_1.select_action(t2)
            t1, t2 = env.step((action_0, action_1))

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


def evaluate_loop(env, agent_0, agent_1, num_episodes, logging):
    """Run evaluation of agents against environment
    TODO: make this a copy of acme
    """
    print("Evaluating")
    print("-----------------------")

    agents = IndependentLeaners([agent_0, agent_1])

    for _ in range(num_episodes // (env.num_envs + 1)):
        rewards_0, rewards_1 = [], []
        timesteps = env.reset()

        while not timesteps[0].last():
            actions = agents.select_action(timesteps)
            timesteps = env.step(actions)

            rewards_0.append(timesteps[0].reward)
            rewards_1.append(timesteps[1].reward)

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
    evaluate_loop(IteratedPrisonersDilemma(50, 5), agent_0, agent_1, 50, False)
