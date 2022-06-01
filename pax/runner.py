import jax.numpy as jnp
import wandb
import jax.random
import jax

from .env import IteratedPrisonersDilemma
from .strategies import Altruistic, TitForTat, Defect
from .independent_learners import IndependentLearners

# TODO: make these a copy of acme


def train_loop(env, agents, num_episodes, watchers, key):
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

            # book keeping
            t = t_prime

            # logging
            if watchers:
                # plot against number of training steps
                agents.log(watchers)
                wandb.log(
                    {
                        "train/training_steps": agents.agents[
                            0
                        ].train_steps,  # assumes the DQN agent is player1
                        "train/step_reward/player_1": float(  # this is the avg. reward across envs
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
        agents.agents[0].episodes += 1
        if watchers:
            wandb.log(
                {
                    "episodes": (agents.agents[0].episodes),
                    "train/episode_reward/player_1": float(rewards_0.mean()),
                    "train/episode_reward/player_2": float(rewards_1.mean()),
                }
            )
    return agents


def evaluate_loop(env, agents, num_episodes, watchers, key):
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

            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "eval/evaluation_steps": agents.agents[0].eval_steps,
                        "eval/step_reward/player_1": float(
                            jnp.array(r_0).mean()
                        ),
                        "eval/step_reward/player_2": float(
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
                    "episodes": (agents.agents[0].episodes),
                    "eval/episode_reward/player_1": float(rewards_0.mean()),
                    "eval/episode_reward/player_2": float(rewards_1.mean()),
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

    # 50 episodes
    evaluate_loop(env, agents, 50, None)
    # 2 episodes of training?
    train_loop(env, agents, 2, [log_print, log_print])
