import logging
import os
import random
import numpy as np
import wandb
import hydra
import jax

from pax.utils import Section
from .env import IteratedPrisonersDilemma
from .independent_learners import IndependentLearners
from .runner import evaluate_loop, train_loop
from .sac.agent import SAC
from .dqn.agent import default_agent
from .dqn.replay_buffer import Replay
from .watchers import policy_logger, policy_logger_dqn, value_logger, value_logger_dqn
from .strategies import TitForTat, Defect, Altruistic, Random, Human

def global_setup(args):
    '''Set up global variables.'''
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
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
    '''Set up env variables.'''
    # Why is it not batch invariant? 
    train_env = IteratedPrisonersDilemma(args.episode_length, args.num_envs)
    test_env = IteratedPrisonersDilemma(args.episode_length, 1)
    return train_env, test_env

def agent_setup(args, logger):
    '''Set up agent variables.'''

    def get_SAC_agent(): 
        sac_agent = SAC(
            state_dim=5,
            action_dim=2,
            discount=args.discount,
            lr=args.lr,
            seed=args.seed,
            )
        return sac_agent

    def get_DQN_agent(): 
        env = IteratedPrisonersDilemma(args.episode_length, args.num_envs)
        dqn_agent = default_agent(
            args, 
            obs_spec=env.observation_spec(),
            action_spec=env.action_spec()
            )
        return dqn_agent

    strategies = {'TitForTat': TitForTat(), 
                'Defect': Defect(), 
                'Altruistic': Altruistic(),
                'Human': Human(), 
                'Random': Random(args.seed),
                'SAC': get_SAC_agent(),
                'DQN':get_DQN_agent() 
                }

    assert args.agent1 in strategies
    assert args.agent2 in strategies

    agent_0 = strategies[args.agent1]
    agent_1 = strategies[args.agent2]
    
    logger.info(f"Agent Pair: {args.agent1} | {args.agent2}")

    return IndependentLearners([agent_0, agent_1])


def watcher_setup(args, logger):
    '''Set up watcher variables.'''
    # TODO: add logging for dqn

    def sac_log(agent):
        policy_dict = policy_logger(agent)
        value_dict = value_logger(agent)
        policy_dict.update(value_dict)
        wandb.log(policy_dict)
        return
    
    def dqn_log(agent):
        policy_dict = policy_logger_dqn(agent)
        value_dict = value_logger_dqn(agent)
        policy_dict.update(value_dict)
        wandb.log(policy_dict)
        return

    def dumb_log(agent):
        return

    strategies = {'TitForTat': dumb_log, 
                  'Defect': dumb_log, 
                  'Altruistic': dumb_log, 
                  'Human': dumb_log, 
                  'Random': dumb_log, 
                  'SAC': sac_log, 
                  'DQN': dqn_log
                }

    assert args.agent1 in strategies
    assert args.agent2 in strategies

    agent_0_log = strategies[args.agent1]
    agent_1_log = strategies[args.agent2]

    return [agent_0_log, agent_1_log]

@hydra.main(config_path="conf", config_name="config")
def main(args):
    '''Set up main.'''
    logger = logging.getLogger() 
    # Adds a bunch of print statements when stuff happens
    with Section("Global setup", logger=logger):
        global_setup(args)

    with Section("Env setup", logger=logger):
        train_env, test_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent_pair = agent_setup(args, logger)

    with Section("Watcher setup", logger=logger):
        watchers = watcher_setup(args, logger)

    train_episodes = args.num_episodes
    eval_every = args.eval_every
    key = jax.random.PRNGKey(args.seed)
    assert not train_episodes % eval_every
    if not args.wandb.log: 
        watchers = False
    # 10,000 / 500 = 20 loops 
    for _ in range(int(train_episodes // eval_every)):
        key, key2 = jax.random.split(key)
        # evaluate for 1 episode each with 100 steps
        evaluate_loop(test_env, agent_pair, 1, watchers, key)
        # train for 500 episodes each with 100 steps
        train_loop(train_env, agent_pair, eval_every, watchers, key2)
    
if __name__ == "__main__":
    main()
