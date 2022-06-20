import time

from pax.ppo.batched_envs import BatchedEnvs
from pax.ppo.ppo import make_agent
from pax.utils import Section

from dm_env import TimeStep
import gym
import hydra
import jax.numpy as jnp
import logging
import wandb


class CartPoleRunner:
    def __init__(self):
        self.global_steps = 0
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()

    def training(self, envs, agent, args):
        """Simple training loop"""
        print("Training ")
        print("-----------------------")
        sequence_length = (
            args.num_steps + 1
        )  # Rollout an addtional step to bootstrap

        # Initialize state and returns
        t = envs.reset()
        episodic_returns = jnp.zeros(shape=(args.num_envs))

        for step in range(0, sequence_length):
            # Update counters
            self.global_steps += 1 * args.num_envs
            self.train_steps += 1 * args.num_envs

            # Select actions and step into environment
            actions = agent.select_action(t)
            t_prime = envs.step(actions)
            episodic_returns = episodic_returns.at[:].add(t_prime.reward)

            # Cartpole is not batched, so need to check termination condition for
            # each environment.
            for i, done in enumerate(t_prime.step_type):
                if done:
                    self.train_episodes += 1
                    print(
                        f"steps={self.global_steps}, return={episodic_returns[i]}"
                    )
                    # Log information
                    if args.wandb.log:
                        wandb.log(
                            {
                                "global_steps": self.global_steps,
                                "env_episodes": self.train_episodes,
                                "train/episode_reward": float(
                                    episodic_returns[i]
                                ),
                                "time_elapsed_minutes": (
                                    time.time() - self.start_time
                                )
                                / 60,
                                "time_elapsed_seconds": time.time()
                                - self.start_time,
                            }
                        )

                    # Reset the episodic return for the ith environment
                    episodic_returns = episodic_returns.at[i].set(0)

                # Truncate returns to penultimate step in sequence length
                # NOTE: sequence length = num_steps + 1
                elif step == args.num_steps - 1:
                    self.train_episodes += 1
                    print(
                        f"steps={self.global_steps}, return={episodic_returns[i]} [Truncated]"
                    )
                    # Log information
                    if args.wandb.log:
                        wandb.log(
                            {
                                "global_steps": self.global_steps,
                                "env_episodes": self.train_episodes,
                                "train/episode_reward": float(
                                    episodic_returns[i]
                                ),
                            }
                        )

            # Update the agent
            agent.update(t, actions, t_prime)

            # Final step of the sequence length.
            if step == args.num_steps:
                # Log information
                if args.wandb.log:
                    wandb.log(
                        {
                            "sgd_steps": agent._logger.metrics["sgd_steps"],
                            "train/loss_total": float(
                                agent._logger.metrics["loss_total"]
                            ),
                            "train/loss_policy": float(
                                agent._logger.metrics["loss_policy"]
                            ),
                            "train/loss_value": float(
                                agent._logger.metrics["loss_value"]
                            ),
                            "train/loss_entropy": float(
                                agent._logger.metrics["loss_entropy"]
                            ),
                        }
                    )

            # Bookkeeping
            t = t_prime

    def evaluation(self, env, agent, args):
        """Simple evaluation loop"""
        print()
        print("Evaluation")
        print("--------------------")
        # Initialize state and returns
        t = env.reset()
        episodic_returns = jnp.zeros(shape=(1))
        eval_done = False

        for _step in range(args.num_steps):
            self.eval_steps += 1
            # Select actions and step into environment
            actions = agent.select_action(t)
            t_prime = env.step(actions)
            episodic_returns = episodic_returns.at[:].add(t_prime.reward)

            # Cartpole is not batched, so need to check termination condition for
            # each environment.
            for i, done in enumerate(t_prime.step_type):
                if done:
                    self.eval_episodes += 1
                    print(
                        f"Episode = {self.eval_episodes}, return={episodic_returns[i]}"
                    )
                    if args.wandb.log:
                        wandb.log(
                            {
                                "eval_episodes": self.eval_episodes,
                                "global_steps": self.global_steps,
                                "eval/episodic_returns": float(
                                    episodic_returns[i]
                                ),
                            }
                        )
                    eval_done = True
                    break
            if eval_done:
                break

            # Bookkeeping
            t = t_prime
        print()


def global_setup(args):
    """Set up global variables."""
    if args.wandb.log:
        print("name", str(args.wandb.name))
        wandb.init(
            reinit=True,
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            config=vars(args),
        )


def env_setup(args, logger):
    eval_env = BatchedEnvs(1, args.seed - 1, args.env_id)
    train_envs = BatchedEnvs(args.num_envs, args.seed, args.env_id)
    logger.info(
        f"Training environments single observation space: {train_envs.observation_spec().num_values}"
    )
    logger.info(
        f"Training environments single action space: { train_envs.action_spec().num_values}"
    )
    return train_envs, eval_env


def agent_setup(args, logger):
    # dummy environment to get the specs
    dummy_env = gym.make(args.env_id)
    dummy_player_id = 0
    seed = args.seed
    agent = make_agent(
        args,
        obs_spec=dummy_env.observation_space.shape,
        action_spec=dummy_env.action_space.n,
        seed=seed,
        player_id=dummy_player_id,
    )
    return agent


@hydra.main(config_path="conf/experiment", config_name="ppo_cartpole")
def main(args):
    logger = logging.getLogger()
    if args.wandb.log:
        wandb.init(
            entity="ucl-dark",
            project="ipd",
            group="cartpole",
            name="run-0",
        )

    with Section("Global setup", logger=logger):
        global_setup(args)

    with Section("Env setup", logger=logger):
        train_envs, eval_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent = agent_setup(args, logger)

    runner = CartPoleRunner()
    num_updates = int(args.total_timesteps // (args.num_envs * args.num_steps))
    for update in range(num_updates):
        print(f"Update {update+1}/{num_updates}")
        runner.training(train_envs, agent, args)
        runner.evaluation(eval_env, agent, args)


if __name__ == "__main__":
    main()
