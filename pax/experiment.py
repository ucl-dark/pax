import logging
import os

import hydra
import omegaconf

import wandb
from pax.env import SequentialMatrixGame
from pax.hyper.ppo import make_hyper
from pax.independent_learners import IndependentLearners
from pax.meta_env import CoinGame, InfiniteMatrixGame, MetaFiniteGame
from pax.naive.naive import make_naive_pg
from pax.naive_exact import NaiveExact
from pax.ppo.ppo import make_agent
from pax.ppo.ppo_gru import make_gru_agent
from pax.runner import Runner
from pax.strategies import (
    Altruistic,
    Defect,
    GrimTrigger,
    Human,
    HyperAltruistic,
    HyperDefect,
    HyperTFT,
    Random,
    TitForTat,
)
from pax.utils import Section
from pax.watchers import (
    logger_hyper,
    logger_naive,
    losses_naive,
    losses_ppo,
    naive_pg_losses,
    policy_logger_ppo,
    policy_logger_ppo_with_memory,
    value_logger_ppo,
)


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
            config=omegaconf.OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            ),
        )


def payoff_setup(args, logger):
    """Set up payoff"""
    games = {
        "ipd": [[-1, -1], [-3, 0], [0, -3], [-2, -2]],
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
                f"Env Type: Regular | Episode Length: {args.num_steps}"
            )

    elif args.env_type == "meta":
        train_env = MetaFiniteGame(
            args.num_envs,
            args.payoff,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
        )
        test_env = MetaFiniteGame(
            args.num_envs,
            args.payoff,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
        )
        if logger:
            logger.info(f"Env Type: Meta | Episode Length: {args.num_steps}")

    elif args.env_type == "coin_game":
        train_env = CoinGame(
            args.num_envs,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        test_env = CoinGame(
            args.num_envs,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        if logger:
            logger.info(
                f"Env Type: CoinGame | Episode Length: {args.num_steps}"
            )

    else:
        train_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
            args.seed,
        )
        test_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
            args.seed + 1,
        )
        if logger:
            logger.info(
                f"Game Type: Infinite | Inner Discount: {args.env_discount}"
            )

    return train_env, test_env


def runner_setup(args):
    return Runner(args)


def agent_setup(args, logger):
    """Set up agent variables."""

    def get_PPO_memory_agent(seed, player_id):
        # dummy environment to get observation and action spec

        if args.env_type == "coin_gmae":
            dummy_env = CoinGame(
                args.num_envs, args.num_steps, args.num_steps, 0
            )
            obs_spec = dummy_env.observation_spec().shape
        else:
            dummy_env = SequentialMatrixGame(
                args.num_envs, args.payoff, args.num_steps
            )
            obs_spec = (dummy_env.observation_spec().num_values,)

        if args.env_type == "meta":
            has_sgd_jit = False
        else:
            has_sgd_jit = True
        ppo_memory_agent = make_gru_agent(
            args,
            obs_spec=obs_spec,
            action_spec=dummy_env.action_spec().num_values,
            seed=seed,
            player_id=player_id,
            has_sgd_jit=has_sgd_jit,
        )
        return ppo_memory_agent

    def get_PPO_agent(seed, player_id):
        # dummy environment to get observation and action spec
        if args.env_type == "coin_game":
            dummy_env = CoinGame(
                args.num_envs, args.num_steps, args.num_steps, 0
            )
            obs_spec = dummy_env.observation_spec().shape
        else:
            dummy_env = SequentialMatrixGame(
                args.num_envs, args.payoff, args.num_steps
            )
            obs_spec = (dummy_env.observation_spec().num_values,)

        if args.env_type == "meta":
            has_sgd_jit = False
        else:
            has_sgd_jit = True

        ppo_agent = make_agent(
            args,
            obs_spec=obs_spec,
            action_spec=dummy_env.action_spec().num_values,
            seed=seed,
            player_id=player_id,
            has_sgd_jit=has_sgd_jit,
        )

        return ppo_agent

    def get_hyper_agent(seed, player_id):
        dummy_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
            args.seed,
        )

        hyper_agent = make_hyper(
            args,
            obs_spec=(dummy_env.observation_spec().num_values,),
            action_spec=dummy_env.action_spec().shape[1],
            seed=seed,
            player_id=player_id,
        )
        return hyper_agent

    def get_naive_pg(seed, player_id):
        if args.env_type == "coin_game":
            dummy_env = CoinGame(
                args.num_envs, args.num_steps, args.num_steps, 0
            )
            obs_spec = dummy_env.observation_spec().shape
        else:
            dummy_env = SequentialMatrixGame(
                args.num_envs, args.payoff, args.num_steps
            )
            obs_spec = (dummy_env.observation_spec().num_values,)

        naive_agent = make_naive_pg(
            args,
            obs_spec=obs_spec,
            action_spec=dummy_env.action_spec().num_values,
            seed=seed,
            player_id=player_id,
        )
        return naive_agent

    def get_naive_learner(seed, player_id):
        dummy_env = InfiniteMatrixGame(
            args.num_envs,
            args.payoff,
            args.num_steps,
            args.env_discount,
            args.seed,
        )

        agent = NaiveExact(
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
        "PPO": get_PPO_agent,
        "PPO_memory": get_PPO_memory_agent,
        "Naive": get_naive_pg,
        # HyperNetworks
        "Hyper": get_hyper_agent,
        "NaiveEx": get_naive_learner,
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

    if args.agent1 == "PPO_memory":
        logger.info(f"PPO with memory: {args.ppo.with_memory}")
    logger.info(f"Agent Pair: {args.agent1} | {args.agent2}")
    logger.info(f"Agent seeds: {seeds[0]} | {seeds[1]}")

    return IndependentLearners([agent_0, agent_1], args)


def watcher_setup(args, logger):
    """Set up watcher variables."""

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

    def naive_pg_log(agent):
        losses = naive_pg_losses(agent)
        policy = policy_logger_ppo(agent)
        value = value_logger_ppo(agent)
        losses.update(value)
        losses.update(policy)
        # if args.wandb.log:
        # wandb.log(losses)
        return

    strategies = {
        "TitForTat": dumb_log,
        "Defect": dumb_log,
        "Altruistic": dumb_log,
        "Human": dumb_log,
        "Random": dumb_log,
        "Grim": dumb_log,
        "PPO": ppo_log,
        "PPO_memory": ppo_log,
        "Naive": naive_pg_log,
        "Hyper": hyper_log,
        "NaiveEx": naive_logger,
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
        runner = runner_setup(args)

    # num episodes
    total_num_ep = int(args.total_timesteps / args.num_steps)
    train_num_ep = int(args.eval_every / args.num_steps)

    print(f"Number of training episodes = {total_num_ep}")
    print(f"Evaluating every {train_num_ep} episodes")
    if not args.wandb.log:
        watchers = False
    for num_update in range(int(total_num_ep // train_num_ep)):
        print(f"Update: {num_update+1}/{int(total_num_ep // train_num_ep)}")
        print()
        runner.train_loop(train_env, agent_pair, train_num_ep, watchers)
        # runner.evaluate_loop(test_env, agent_pair, 1, watchers)


if __name__ == "__main__":
    main()
