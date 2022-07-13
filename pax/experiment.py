import logging
import os

from pax.dqn.agent import default_agent
from pax.env import (
    InfiniteMatrixGame,
    SequentialMatrixGame,
)
from pax.hyper.ppo import make_hyper
from pax.hyper.ppo_gru import make_gru_hyper
from pax.independent_learners import IndependentLearners
from pax.ppo.ppo import make_agent
from pax.ppo.ppo_gru import make_gru_agent
from pax.runner import Runner
from pax.sac.agent import SAC
from pax.strategies import (
    HyperAltruistic,
    HyperDefect,
    HyperTFT,
    NaiveLearner,
    TitForTat,
    Defect,
    Altruistic,
    Random,
    Human,
    GrimTrigger,
)
from pax.utils import Section
from pax.watchers import (
    logger_naive,
    losses_naive,
    policy_logger,
    policy_logger_dqn,
    logger_hyper,
    policy_logger_hyper_gru,
    value_logger,
    value_logger_dqn,
    losses_ppo,
    policy_logger_ppo,
    value_logger_ppo,
    policy_logger_ppo_with_memory,
)

import hydra
import wandb


def global_setup(args):
    """Set up global variables."""
    os.makedirs(args.save_dir, exist_ok=True)
    if args.wandb.log:
        print("name", str(args.wandb.name))
        if args.debug:
            args.wandb.group = "debug-" + args.wandb.group
        wandb.init(
            reinit=True,
            entity=str(args.wandb.entity),
            project=str(args.wandb.project),
            group=str(args.wandb.group),
            name=str(args.wandb.name),
            config=vars(args),
        )


def payoff_setup(args, logger):
    """Set up payoff"""
    games = {
        "ipd": [[2, 2], [0, 3], [3, 0], [1, 1]],
        "stag": [[4, 4], [1, 3], [3, 1], [2, 2]],
        "sexes": [[3, 2], [0, 0], [0, 0], [2, 3]],
        "chicken": [[0, 0], [-1, 1], [1, -1], [-2, -2]],
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

        payoff = args.payoff
        logger.info(f"Game: Custom | payoff: {payoff}")

    else:
        assert args.game in games, f"{args.game} not in {games.keys()}"
        args.payoff = games[args.game]
        logger.info(f"Game: {args.game} | payoff: {args.payoff}")


def env_setup(args, logger=None):
    """Set up env variables."""
    payoff_setup(args, logger)
    if args.env_type == "finite":
        train_env = SequentialMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
        )
        test_env = SequentialMatrixGame(1, args.payoff, args.num_steps)
        if logger:
            logger.info(
                f"Game Type: Finite | Episode Length: {args.num_steps}"
            )

    else:
        train_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
        )
        test_env = InfiniteMatrixGame(
            args.num_envs, args.payoff, args.num_steps, args.env_discount
        )
        if logger:
            logger.info(
                f"Game Type: Infinite | Inner Discount: {args.env_discount}"
            )

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
            args.num_envs, args.payoff, args.num_steps
        )
        dqn_agent = default_agent(
            args,
            obs_spec=dummy_env.observation_spec(),
            action_spec=dummy_env.action_spec(),
            seed=seed,
            player_id=player_id,
        )
        return dqn_agent

    def get_PPO_memory_agent(seed, player_id):
        # dummy environment to get observation and action spec
        dummy_env = SequentialMatrixGame(
            args.num_envs, args.payoff, args.num_steps
        )
        ppo_memory_agent = make_gru_agent(
            args,
            obs_spec=(dummy_env.observation_spec().num_values,),
            action_spec=dummy_env.action_spec().num_values,
            seed=seed,
            player_id=player_id,
        )
        return ppo_memory_agent

    def get_PPO_agent(seed, player_id):
        # dummy environment to get observation and action spec
        dummy_env = SequentialMatrixGame(
            args.num_envs, args.payoff, args.num_steps
        )
        if args.ppo.with_memory:
            ppo_agent = make_gru_agent(
                args,
                obs_spec=(dummy_env.observation_spec().num_values,),
                action_spec=dummy_env.action_spec().num_values,
                seed=seed,
                player_id=player_id,
            )
        else:
            ppo_agent = make_agent(
                args,
                obs_spec=(dummy_env.observation_spec().num_values,),
                action_spec=dummy_env.action_spec().num_values,
                seed=seed,
                player_id=player_id,
            )
        return ppo_agent

    def get_hyper_agent(seed, player_id):
        dummy_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
        )

        if args.ppo.with_memory:
            hyper_agent = make_gru_hyper(
                args,
                obs_spec=(dummy_env.observation_spec().num_values,),
                action_spec=dummy_env.action_spec().shape[1],
                seed=seed,
                player_id=player_id,
            )
        else:
            hyper_agent = make_hyper(
                args,
                obs_spec=(dummy_env.observation_spec().num_values,),
                action_spec=dummy_env.action_spec().shape[1],
                seed=seed,
                player_id=player_id,
            )
        return hyper_agent

    def get_naive_learner(seed, player_id):
        dummy_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
        )

        agent = NaiveLearner(
            action_dim=dummy_env.action_spec().shape[1],
            env=dummy_env,
            lr=args.naive.lr,
            seed=seed,
            player_id=player_id,
        )
        return agent

    strategies = {
        "TitForTat": TitForTat,
        "Defect": Defect,
        "Altruistic": Altruistic,
        "Human": Human,
        "Random": Random,
        "Grim": GrimTrigger,
        "SAC": get_SAC_agent,
        "DQN": get_DQN_agent,
        "PPO": get_PPO_agent,
        "PPO_memory": get_PPO_memory_agent,
        # HyperNetworks
        "Hyper": get_hyper_agent,
        "NaiveLearner": get_naive_learner,
        "HyperAltruistic": HyperAltruistic,
        "HyperDefect": HyperDefect,
        "HyperTFT": HyperTFT,
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

    logger.info(f"PPO with memory: {args.ppo.with_memory}")
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

    def ppo_log(agent):
        losses = losses_ppo(agent)
        if args.ppo.with_memory:
            policy = policy_logger_ppo_with_memory(agent)
        else:
            policy = policy_logger_ppo(agent)
            value = value_logger_ppo(agent)
            losses.update(value)
        losses.update(policy)
        if args.wandb.log:
            wandb.log(losses)
        return

    def dumb_log(agent, *args):
        return

    def hyper_log(agent):
        losses = losses_ppo(agent)
        if args.ppo.with_memory:
            policy = policy_logger_hyper_gru(agent)
        else:
            policy = logger_hyper(agent)
        losses.update(policy)
        if args.wandb.log:
            wandb.log(losses)
        return

    def naive_logger(agent):
        losses = losses_naive(agent)
        policy = logger_naive(agent)
        losses.update(policy)
        if args.wandb.log:
            wandb.log(losses)
        return

    strategies = {
        "TitForTat": dumb_log,
        "Defect": dumb_log,
        "Altruistic": dumb_log,
        "Human": dumb_log,
        "Random": dumb_log,
        "Grim": dumb_log,
        "SAC": sac_log,
        "DQN": dqn_log,
        "PPO": ppo_log,
        "PPO_memory": ppo_log,
        "Hyper": hyper_log,
        "NaiveLearner": naive_logger,
        "HyperAltruistic": dumb_log,
        "HyperDefect": dumb_log,
        "HyperTFT": dumb_log,
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

    # num episodes
    total_num_ep = int(args.total_timesteps / (args.num_steps))
    train_num_ep = int(args.eval_every / (args.num_steps))

    print(f"Number of training episodes = {total_num_ep}")
    print(f"Evaluating every {train_num_ep} episodes")
    if not args.wandb.log:
        watchers = False
    for num_update in range(int(total_num_ep // train_num_ep)):
        print(f"Update: {num_update}/{int(total_num_ep // train_num_ep)}")
        print()

        runner.evaluate_loop(test_env, agent_pair, args.num_envs, watchers)
        runner.train_loop(train_env, agent_pair, train_num_ep, watchers)


if __name__ == "__main__":
    main()
