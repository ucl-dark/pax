import logging
import os
import random
import numpy as np
import wandb
import hydra
import jax

from pax.utils import Section
from pax.env import (
    SequentialMatrixGame,
    IteratedPrisonersDilemma,
    StagHunt,
    BattleOfTheSexes,
    Chicken,
)
from pax.independent_learners import IndependentLearners
from pax.runner import Runner
from pax.sac.agent import SAC
from pax.dqn.agent import default_agent
from pax.dqn.replay_buffer import Replay
from pax.watchers import (
    policy_logger,
    policy_logger_dqn,
    value_logger,
    value_logger_dqn,
)
from pax.strategies import TitForTat, Defect, Altruistic, Random, Human


def global_setup(args):
    """Set up global variables."""
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
    """Set up env variables."""
    games = {
        "ipd": IteratedPrisonersDilemma,
        "stag": StagHunt,
        "sexes": BattleOfTheSexes,
        "chicken": Chicken,
    }
    if args.payoff is not None:
        assert (
            len(args.payoff) == 4
        ), f"Expected length 4 but got {len(args.payoff)}"
        assert (
            len(args.payoff[0]) == 2
        ), f"Expected length 2 but got {len(args.payoff[0])}"
        assert (
            len(args.payoff[1]) == 2
        ), f"Expected length 2 but got {len(args.payoff[1])}"
        assert (
            len(args.payoff[2]) == 2
        ), f"Expected length 2 but got {len(args.payoff[2])}"
        assert (
            len(args.payoff[3]) == 2
        ), f"Expected length 2 but got {len(args.payoff[3])}"
        train_env = SequentialMatrixGame(
            args.episode_length, args.num_envs, args.payoff
        )
        test_env = SequentialMatrixGame(args.episode_length, 1, args.payoff)
        logger.info(f"Game: Custom | payoff: {args.payoff}")
    else:
        assert args.game in games, f"{args.game} not in {games.keys()}"
        train_env = games[args.game](args.episode_length, args.num_envs)
        test_env = games[args.game](args.episode_length, 1)
        logger.info(f"Game: {args.game} | payoff: {train_env.payoff}")

    return train_env, test_env


def runner_setup():
    return Runner()


def agent_setup(args, logger):
    """Set up agent variables."""

    def get_SAC_agent(*args):
        sac_agent = SAC(
            state_dim=5,
            action_dim=2,
            discount=args.discount,
            lr=args.lr,
            seed=args.seed,
        )
        return sac_agent

    def get_DQN_agent(seed, player_id):
        # dummy environment to get observation and action spec
        dummy_env = SequentialMatrixGame(
            args.episode_length, args.num_envs, args.payoff
        )
        dqn_agent = default_agent(
            args,
            obs_spec=dummy_env.observation_spec(),
            action_spec=dummy_env.action_spec(),
            seed=seed,
            player_id=player_id,
        )
        return dqn_agent

    strategies = {
        "TitForTat": TitForTat,
        "Defect": Defect,
        "Altruistic": Altruistic,
        "Human": Human,
        "Random": Random,
        "SAC": get_SAC_agent,
        "DQN": get_DQN_agent,
    }

    assert args.agent1 in strategies
    assert args.agent2 in strategies

    # TODO: Is there a better way to start agents with different seeds?
    num_agents = 2
    seeds = [seed for seed in range(args.seed, args.seed + num_agents)]
    # Create Player IDs by normalizing seeds to 1, 2 respectively
    pids = [
        seed % seed + i if seed != 0 else 1
        for seed, i in zip(seeds, range(1, num_agents + 1))
    ]
    agent_0 = strategies[args.agent1](seeds[0], pids[0])  # player 1
    agent_1 = strategies[args.agent2](seeds[1], pids[1])  # player 2

    logger.info(f"Agent Pair: {args.agent1} | {args.agent2}")
    logger.info(f"Agent seeds: {seeds[0]} | {seeds[1]}")

    return IndependentLearners([agent_0, agent_1])


def watcher_setup(args, logger):
    """Set up watcher variables."""

    def sac_log(agent, *args):
        policy_dict = policy_logger(agent)
        value_dict = value_logger(agent)
        policy_dict.update(value_dict)
        wandb.log(policy_dict)
        return

    def dqn_log(agent):
        policy_dict = policy_logger_dqn(agent)
        value_dict = value_logger_dqn(agent)
        policy_dict.update(value_dict)
        if args.wandb.log:
            wandb.log(policy_dict)
        return

    def dumb_log(agent, *args):
        return

    strategies = {
        "TitForTat": dumb_log,
        "Defect": dumb_log,
        "Altruistic": dumb_log,
        "Human": dumb_log,
        "Random": dumb_log,
        "SAC": sac_log,
        "DQN": dqn_log,
    }

    assert args.agent1 in strategies
    assert args.agent2 in strategies

    agent_0_log = strategies[args.agent1]
    agent_1_log = strategies[args.agent2]

    return [agent_0_log, agent_1_log]


@hydra.main(config_path="conf", config_name="config")
def main(args):
    """Set up main."""
    logger = logging.getLogger()
    with Section("Global setup", logger=logger):
        global_setup(args)

    with Section("Env setup", logger=logger):
        train_env, test_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent_pair = agent_setup(args, logger)

    with Section("Watcher setup", logger=logger):
        watchers = watcher_setup(args, logger)

    with Section("Runner setup", logger=logger):
        runner = runner_setup()

    # TODO: Do we need this?
    # assert not train_episodes % eval_every
    key = jax.random.PRNGKey(args.seed)
    for _ in range(int(args.num_episodes // args.eval_every)):
        key, key1, key2 = jax.random.split(key, num=3)
        runner.evaluate_loop(test_env, agent_pair, 1, watchers, key1)
        runner.train_loop(
            train_env, agent_pair, args.eval_every, watchers, key2
        )


if __name__ == "__main__":
    main()
