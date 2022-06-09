import logging
import gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from pax.utils import Section
from pax.ppo.ppo import PPO, make_agent
from pax.ppo.networks import critic_network
import wandb
import time


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
    # observation = env.reset()
    env.close()


def train(agent, envs, args):
    """Train agent"""
    # NOTE: Each policy rollout is of size: num_envs * num_steps. Default=4*128=512\
    key = jax.random.PRNGKey(args.seed)
    key, key1 = jax.random.split(key)

    # TODO: move this to the buffer
    obs = jnp.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    )
    actions = jnp.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    )
    logprobs = jnp.zeros((args.num_steps, args.num_envs))
    rewards = jnp.zeros((args.num_steps, args.num_envs))
    dones = jnp.zeros((args.num_steps, args.num_envs))
    values = jnp.zeros((args.num_steps, args.num_envs))

    global_step = 0
    # start_time = time.time()
    next_obs = envs.reset()
    next_done = jnp.zeros(args.num_envs)
    batch_size = int(args.num_envs * args.num_steps)
    num_updates = args.total_timesteps // batch_size
    # minibatch_size = int(args.batch_size // args.num_minibatches)

    # TODO: Layout of training
    # initialize rewards storage
    # reset environment
    # for each time step
    ## select an action
    ## take a step in the environment
    ## append rewards
    ## update agent
    ## set next timestep as current timestep
    ## log

    for _update in range(1, num_updates + 1):
        # TODO: Go through a number of steps * num envs of steps = batch size
        for step in range(0, args.num_steps):
            # Increase number of steps depending on number of environments
            key, key1 = jax.random.split(key)
            global_step += 1 * args.num_envs
            obs = obs.at[step].set(next_obs)
            dones = dones.at[step].set(next_done)

            # This is the forward pass
            action, logprob, _, value = agent.get_action_and_value(
                key1, next_obs, action=None
            )

            values = values.at[step].set(value.flatten())
            actions = actions.at[step].set(action)
            logprobs = logprobs.at[step].set(logprob)

            # TODO: Sketchy to convert it to a numpy array
            next_obs, reward, done, info = envs.step(np.array(action))
            rewards = rewards.at[step].set(reward)

            for item in info:
                if "episode" in item.keys():
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}"
                    )

            # gae
            # TODO: make this the better version later
            # next_value = agent._critic_forward(agent._state.critic_params, next_obs).reshape(1, -1)
            # if args.gae:
            #     advantages =
            # returns = jnp.zeros_like(rewards)
            # rewards, discounts, values
            # for t in reversed(range(args.num_steps)):
            #     if t == args.num_steps - 1:
            #         nextnonterminal = 1.0 - next_done
            #         next_return = next_value
            #     else:
            #         nextnonterminal = 1.0 - dones[t+1]
            #         next_return = returns[t+1]
            #     print("returns.shape", returns.shape)
            #     # print("returns", returns)
            #     print(returns[t].shape)
            #     x = rewards[t] + args.gamma * next_return
            #     print(x.shape)
            #     returns = returns.at[t].set(rewards[t] + args.gamma * nextnonterminal * next_return)
            #     # returns[t] = rewards[t] + args.gamma * next_return
            # advantages = returns - values
        # TODO: Need an array of shape
        # End of this for loop, I have all experience from this update

    # for _ in range(200):
    #     action = envs.action_space.sample()
    #     print("in for second loop")
    #     print("action", action)
    #     print("action.shape", action.shape)
    #     observation, reward, done, info = envs.step(action)

    #     for item in info:
    #         if "episode" in item.keys():
    #             print(f"episodic return: {item['episode']['r']}")
    #             # NOTE: there is no `observation = env.reset()` anymore
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
