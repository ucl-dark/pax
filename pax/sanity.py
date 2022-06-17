from dm_env import TimeStep

import logging
import gym
import hydra
import jax.numpy as jnp
import wandb
import time

from pax.ppo.batched_envs import BatchedEnvs
from pax.ppo.ppo import PPO, make_agent
from pax.utils import Section
from bsuite.utils import gym_wrapper


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
    train_envs = BatchedEnvs(args.num_envs, args.seed, args.env_id)
    logger.info(
        f"Training environments single observation space: {train_envs.observation_spec().num_values}"
    )
    logger.info(
        f"Training environments single action space: { train_envs.action_spec().num_values}"
    )
    return train_envs


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


def evaluate(agent, env, args):
    """Evaluate agent"""
    # TODO: create a runner class to store all relevant variables
    # print()
    # print("Evaluation")
    # print("--------------------")
    # done = False
    # env = BatchedEnvs(1, args.seed, args.env_id)
    # num_steps = 500
    # t = env.reset()
    # eval_episodic_returns = jnp.zeros(shape=(1))
    # for _step in range(num_steps):
    #     actions = agent.select_action(t)
    #     t_prime = env.step(actions)
    #     eval_episodic_returns = eval_episodic_returns.at[:].add(t_prime.reward)

    #     # Cartpole specific logic where an episode can terminate mid-rollout
    #     # Check if any of the environments have terminated
    #     for i, done in enumerate(t_prime.step_type):
    #         # Not done = 0, done = 1
    #         if done:
    #             eval_episodes += 1
    #             print(
    #                 f"Evaluation episode = {eval_episodes}, episodic_return={eval_episodic_returns[i]}"
    #             )
    #             if args.wandb.log:
    #                 wandb.log(
    #                     {
    #                         "Evaluation Episode": eval_episodes,
    #                         "global_steps": global_step,
    #                         "eval/episodic_returns": float(
    #                             eval_episodic_returns[i]
    #                         ),
    #                     }
    #                 )
    #             eval_done = True
    #             break
    #     if eval_done:
    #         break

    #     # Bookkeeping
    #     t = t_prime
    # print()
    pass


def train(agent, envs, args):
    """Train agent"""
    global_step = 0
    env_episodes = 0
    eval_episodes = 0
    rollout_steps = (
        args.num_steps + 1
    )  # Rollout an addtional step to bootstrap
    batch_size = int(args.num_envs * args.num_steps)
    num_updates = args.total_timesteps // batch_size
    start_time = time.time()

    # Total number of updates = number of steps / batch_size
    for update in range(1, num_updates + 1):
        print("Training")
        print("--------------------")
        print(f"Update {update}/{num_updates}")

        # Reset environments and initialize episodic returns
        t = envs.reset()
        episodic_returns = jnp.zeros(shape=(args.num_envs))

        # Each policy rollout is of size = num_envs * num_steps+1
        for step in range(0, rollout_steps):
            global_step += 1 * args.num_envs
            actions = agent.select_action(t)
            t_prime = envs.step(actions)
            episodic_returns = episodic_returns.at[:].add(t_prime.reward)

            # Cartpole specific logic where an episode can terminate mid-rollout
            # Check if any of the environments have terminated
            for i, done in enumerate(t_prime.step_type):
                if done:
                    env_episodes += 1
                    print(
                        f"global_step={global_step}, episodic_return={episodic_returns[i]}"
                    )
                    if args.wandb.log:
                        wandb.log(
                            {
                                "global_steps": global_step,
                                "env_episodes": env_episodes,
                                "train/episode_reward": float(
                                    episodic_returns[i]
                                ),
                                "time_elapsed_minutes": (
                                    time.time() - start_time
                                )
                                / 60,
                                "time_elapsed_seconds": time.time()
                                - start_time,
                            }
                        )
                    # Reset the ith environment
                    t_reset = envs.envs[
                        i
                    ].reset()  # NOTE: this is single environment, not multiple

                    # Reset the ith observation with the new_observation
                    t_prime = TimeStep(
                        step_type=t_prime.step_type,
                        observation=t_prime.observation.at[i].set(
                            t_reset.observation
                        ),
                        reward=t_prime.reward,
                        discount=t_prime.discount,
                    )

                    # Reset the episodic return for the ith environment to 0
                    episodic_returns = episodic_returns.at[i].set(0)

                # Report truncated episodes
                elif step == args.num_steps - 1:
                    env_episodes += 1
                    print(
                        f"global_step={global_step}, episodic_return={episodic_returns[i]} (Truncated)"
                    )
                    if args.wandb.log:
                        wandb.log(
                            {
                                "global_steps": global_step,
                                "env_episodes": env_episodes,
                                "train/episode_reward": float(
                                    episodic_returns[i]
                                ),
                            }
                        )

                # Log metrics if SGD has occured
                if args.wandb.log and global_step % (batch_size + 1) == 0:
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

            agent.update(t, actions, agent._infos, t_prime)
            # Bookkeeping
            t = t_prime

        print()
        print("Evaluation")
        print("--------------------")
        eval_done = False
        eval_env = BatchedEnvs(1, args.seed, args.env_id)
        num_eval_steps = 500
        t = eval_env.reset()
        eval_episodic_returns = jnp.zeros(shape=(1))
        for _step in range(num_eval_steps):
            actions = agent.select_action(t)
            t_prime = envs.step(actions)
            eval_episodic_returns = eval_episodic_returns.at[:].add(
                t_prime.reward
            )

            # Cartpole specific logic where an episode can terminate mid-rollout
            # Check if any of the environments have terminated
            for i, done in enumerate(t_prime.step_type):
                # Not done = 0, done = 1
                if done:
                    eval_episodes += 1
                    print(
                        f"Evaluation episode = {eval_episodes}, episodic_return={eval_episodic_returns[i]}"
                    )
                    if args.wandb.log:
                        wandb.log(
                            {
                                "Evaluation Episode": eval_episodes,
                                "global_steps": global_step,
                                "eval/episodic_returns": float(
                                    eval_episodic_returns[i]
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

    envs.close()


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
        train_envs = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent = agent_setup(args, logger)

    train(agent, train_envs, args)


if __name__ == "__main__":
    main()
