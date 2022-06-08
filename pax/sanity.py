import logging
import gym
import hydra
from pax.utils import Section
from pax.ppo.ppo import PPO, make_agent
from pax.ppo.networks import critic_network
import wandb


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
    def make_env(env_id, seed):
        def thunk():
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed)
            return env

        return thunk

    # SyncVectorEnv allows one to stack n environments
    # Setup test environments
    test_env = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed)])
    test_env.action_space.seed(args.seed)
    test_env.observation_space.seed(args.seed)

    # Setup train environments
    train_envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed + i)
            for i in range(1, args.num_envs + 1)
        ]
    )
    train_envs.action_space.seed(args.seed + 1)
    train_envs.observation_space.seed(args.seed + 1)

    # Logging
    logger.info(
        f"Training environments single observation space: {train_envs.single_observation_space.shape}"
    )
    logger.info(
        f"Training environments single action space: { train_envs.single_action_space.n}"
    )
    return train_envs, test_env


def agent_setup(args, logger):
    # dummy environment to get the specs
    dummy_env = gym.make(args.env_id)
    dummy_player_id = 0
    agent = make_agent(
        args,
        obs_spec=dummy_env.observation_space.shape,
        action_spec=dummy_env.action_space.n,
        seed=args.seed,
        player_id=dummy_player_id,
    )
    return agent


def evaluate(agent, env, args):
    """Evaluate agent"""
    observation = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        for item in info:
            if "episode" in item.keys():
                print(f"episodic return: {item['episode']['r']}")
                # NOTE: there is no `observation = env.reset()` anymore
    env.close()


def train(agent, envs, args):
    """Train agent"""
    observation = envs.reset()
    # NOTE: Each policy rollout is of size: num_envs * num_steps. Default=4*128=512
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, done, info = envs.step(action)
        for item in info:
            if "episode" in item.keys():
                print(f"episodic return: {item['episode']['r']}")
                # NOTE: there is no `observation = env.reset()` anymore
    envs.close()


@hydra.main(config_path="conf/experiment", config_name="ppo_cartpole")
def main(args):
    logger = logging.getLogger()
    if args.wandb.log:
        wandb.init(
            entity="ucl-dark",
            project="ipd",
            group="cartpole-debug",
            name="run-0",
        )

    with Section("Global setup", logger=logger):
        global_setup(args)

    with Section("Env setup", logger=logger):
        train_envs, test_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent = agent_setup(args, logger)

    train(agent, train_envs, args)
    evaluate(agent, test_env, args)


if __name__ == "__main__":
    main()
