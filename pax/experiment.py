import logging
import os
import random
import numpy as np
import wandb
import hydra

from pax.utils import Section

from .env import IteratedPrisonersDilemma
from .independent_learners import IndependentLearners
from .runner import evaluate_loop, train_loop
from .sac.agent import SAC
from .watchers import policy_logger, value_logger


def global_setup(args):
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    if args.wandb.log:
        wandb.init(
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            config=vars(args),
        )


def env_setup(args, logger):
    train_env = IteratedPrisonersDilemma(args.episode_length, args.num_envs)
    test_env = IteratedPrisonersDilemma(args.episode_length, 1)
    return train_env, test_env


def agent_setup(args, logger):
    # TODO: make configurable
    agent_0 = SAC(
        state_dim=5,
        action_dim=2,
        discount=args.discount,
        lr=args.lr,
        seed=args.seed,
    )

    agent_1 = SAC(
        state_dim=5,
        action_dim=2,
        discount=args.discount,
        lr=args.lr,
        seed=args.seed,
    )

    logger.info(f"Agent Pair: {agent_0.name} | {agent_1.name} ")

    return IndependentLearners([agent_0, agent_1])


def watcher_setup(args, logger):
    # TODO: make configurable based on previous agents

    def sac_log(agent):
        policy_dict = policy_logger(agent)
        value_dict = value_logger(agent)
        policy_dict.update(value_dict)
        wandb.log(policy_dict)
        return

    def dumb_log(agent):
        return

    return [sac_log, dumb_log]


@hydra.main(config_path="conf", config_name="config")
def main(args):
    logger = logging.getLogger()
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
    assert not train_episodes % eval_every
    for _ in range(int(train_episodes // eval_every)):
        evaluate_loop(test_env, agent_pair, 1, watchers)
        train_loop(train_env, agent_pair, eval_every, watchers)


if __name__ == "__main__":
    main()
