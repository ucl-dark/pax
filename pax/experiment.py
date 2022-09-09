from datetime import datetime
import logging
import os

from evosax import OpenES, CMA_ES, PGPE, ParameterReshaper, SimpleGA
import hydra
import omegaconf
import wandb


from pax.env_inner import InfiniteMatrixGame
from pax.env_inner import SequentialMatrixGame
from pax.env_meta import CoinGame, MetaFiniteGame
from pax.evaluation_ipd import EvalRunnerIPD
from pax.evaluation_cg import EvalRunnerCG
from pax.hyper.ppo import make_hyper
from pax.learners import IndependentLearners, EvolutionaryLearners
from pax.naive.naive import make_naive_pg
from pax.naive_exact import NaiveExact
from pax.ppo.ppo import make_agent
from pax.ppo.ppo_gru import make_gru_agent
from pax.runner_evo import EvoRunner
from pax.runner_rl import Runner
from pax.strategies import (
    Altruistic,
    Defect,
    GreedyCoinChaser,
    GrimTrigger,
    Human,
    HyperAltruistic,
    HyperDefect,
    HyperTFT,
    Random,
    Stay,
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
    save_dir = f"{args.save_dir}/{str(datetime.now()).replace(' ', '_').replace(':', '.')}"
    if not args.eval:
        os.makedirs(
            save_dir,
            exist_ok=True,
        )
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
            settings=wandb.Settings(code_dir="."),
        )
        wandb.run.log_code(".")
    return save_dir


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
    if args.env_type == "sequential":
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
            1,
            args.payoff,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
        )
        if logger:
            if args.evo:
                logger.info(f"Env Type: Meta | Generations: {args.num_steps}")
            logger.info(f"Env Type: Meta | Episode Length: {args.num_steps}")

    elif args.env_type == "coin_game":
        train_env = CoinGame(
            args.num_envs,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
            seed=args.seed,
            cnn=args.ppo.with_cnn,
        )
        test_env = CoinGame(
            1,
            inner_ep_length=args.num_inner_steps,
            num_steps=args.num_steps,
            seed=args.seed,
            cnn=args.ppo.with_cnn,
        )
        if logger:
            logger.info(
                f"Env Type: CoinGame | Episode Length: {args.num_steps}"
            )

    elif args.env_type == "infinite":
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


def runner_setup(args, agents, save_dir, logger):
    if args.eval:
        if args.env_id == "ipd":
            logger.info("Evaluating with EvalRunnerIPD")
            return EvalRunnerIPD(args)
        elif args.env_id == "coin_game":
            logger.info("Evaluating with EvalRunnerCG")
            return EvalRunnerCG(args)

    if args.evo:
        agent1, _ = agents.agents
        algo = args.es.algo
        strategies = {"CMA_ES", "OpenES", "PGPE", "SimpleGA"}
        assert algo in strategies, f"{algo} not in evolution strategies"

        def get_ga_strategy(agent):
            """Returns the SimpleGA strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(agent._state.params)
            strategy = SimpleGA(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize,
            )
            es_params = strategy.default_params
            return strategy, es_params, param_reshaper

        def get_cma_strategy(agent):
            """Returns the CMA strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(agent._state.params)
            strategy = CMA_ES(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize,
                elite_ratio=args.es.elite_ratio,
            )
            es_params = strategy.default_params
            return strategy, es_params, param_reshaper

        def get_openes_strategy(agent):
            """Returns the OpenES strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(agent._state.params)
            strategy = OpenES(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize,
            )
            # Update basic parameters of OpenES strategy
            es_params = strategy.default_params.replace(
                sigma_init=args.es.sigma_init,
                sigma_decay=args.es.sigma_decay,
                sigma_limit=args.es.sigma_limit,
                init_min=args.es.init_min,
                init_max=args.es.init_max,
                clip_min=args.es.clip_min,
                clip_max=args.es.clip_max,
            )

            # Update optimizer-specific parameters of Adam
            es_params = es_params.replace(
                opt_params=es_params.opt_params.replace(
                    lrate_init=args.es.lrate_init,
                    lrate_decay=args.es.lrate_decay,
                    lrate_limit=args.es.lrate_limit,
                    beta_1=args.es.beta_1,
                    beta_2=args.es.beta_2,
                    eps=args.es.eps,
                )
            )
            return strategy, es_params, param_reshaper

        def get_pgpe_strategy(agent):
            """Returns the PGPE strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(agent._state.params)
            strategy = PGPE(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize,
                elite_ratio=args.es.elite_ratio,
            )
            es_params = strategy.default_params
            return strategy, es_params, param_reshaper

        if algo == "CMA_ES":
            strategy, es_params, param_reshaper = get_cma_strategy(agent1)
        elif algo == "OpenES":
            strategy, es_params, param_reshaper = get_openes_strategy(agent1)
        elif algo == "PGPE":
            strategy, es_params, param_reshaper = get_pgpe_strategy(agent1)
        elif algo == "SimpleGA":
            strategy, es_params, param_reshaper = get_ga_strategy(agent1)

        logger.info(f"Evolution Strategy: {algo}")

        logger.info("Training with EvoRunner")
        return EvoRunner(args, strategy, es_params, param_reshaper, save_dir)
    else:
        logger.info("Training with Runner")
        return Runner(args, save_dir)


# flake8: noqa: C901
def agent_setup(args, logger):
    """Set up agent variables."""

    def get_PPO_memory_agent(seed, player_id):
        # dummy environment to get observation and action spec
        if args.env_type == "coin_game":
            dummy_env = CoinGame(
                args.num_envs,
                args.num_steps,
                args.num_steps,
                0,
                args.ppo.with_cnn,
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
                args.num_envs,
                args.num_steps,
                args.num_steps,
                0,
                args.ppo.with_cnn,
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
                args.num_envs,
                args.num_steps,
                args.num_steps,
                0,
                args.ppo.with_cnn,
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

    def get_random_agent(seed, player_id):
        if args.env_type == "coin_game":
            num_actions = (
                CoinGame(
                    args.num_envs,
                    args.num_steps,
                    args.num_steps,
                    0,
                    args.ppo.with_cnn,
                )
                .action_spec()
                .num_values
            )
        else:
            num_actions = (
                SequentialMatrixGame(
                    args.num_envs, args.payoff, args.num_steps
                )
                .action_spec()
                .num_values
            )

        random_agent = Random(num_actions)
        random_agent.player_id = player_id
        return random_agent

    # flake8: noqa: C901
    def get_stay_agent(seed, player_id):
        if args.env_type == "coin_game":
            num_actions = (
                CoinGame(
                    args.num_envs,
                    args.num_steps,
                    args.num_steps,
                    0,
                    args.ppo.with_cnn,
                )
                .action_spec()
                .num_values
            )
        else:
            num_actions = (
                SequentialMatrixGame(
                    args.num_envs, args.payoff, args.num_steps
                )
                .action_spec()
                .num_values
            )

        agent = Stay(num_actions)
        agent.player_id = player_id
        return agent

    strategies = {
        "TitForTat": TitForTat,
        "Defect": Defect,
        "Altruistic": Altruistic,
        "Human": Human,
        "Random": get_random_agent,
        "Stay": get_stay_agent,
        "Grim": GrimTrigger,
        "Greedy": GreedyCoinChaser,
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
        if not args.ppo.with_memory:
            raise ValueError("Can't use PPO_memory but set ppo.memory=False")
    if args.agent1 == "PPO":
        if args.ppo.with_memory:
            raise ValueError(
                "Can't use ppo.memory=False but set agent=PPO_memory"
            )

    if args.agent1 in ["PPO", "PPO_memory"] and args.ppo.with_cnn:
        logger.info(f"PPO with CNN: {args.ppo.with_cnn}")
    logger.info(f"Agent Pair: {args.agent1} | {args.agent2}")
    logger.info(f"Agent seeds: {seeds[0]} | {seeds[1]}")

    if args.evo and not args.eval:
        logger.info("Using EvolutionaryLearners")
        return EvolutionaryLearners([agent_0, agent_1], args)
    return IndependentLearners([agent_0, agent_1], args)


def watcher_setup(args, logger):
    """Set up watcher variables."""

    def ppo_log(agent):
        losses = losses_ppo(agent)
        if not args.env_type == "coin_game":
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
        if not args.env_type == "coin_game":
            policy = policy_logger_ppo(agent)
            value = value_logger_ppo(agent)
            losses.update(value)
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
        "Stay": dumb_log,
        "Grim": dumb_log,
        "Greedy": dumb_log,
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
        save_dir = global_setup(args)

    with Section("Env setup", logger=logger):
        train_env, test_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent_pair = agent_setup(args, logger)

    with Section("Watcher setup", logger=logger):
        watchers = watcher_setup(args, logger)

    with Section("Runner setup", logger=logger):
        runner = runner_setup(args, agent_pair, save_dir, logger)

    if not args.wandb.log:
        watchers = False

    # If evaluating, pass in the number of seeds you want to evaluate over
    if args.eval:
        runner.eval_loop(train_env, agent_pair, args.num_seeds, watchers)

    # If training, get the number of iterations to run

    else:
        # Lets do some warm ups
        print("WarmUps")
        import copy

        agent1_fresh = copy.deepcopy(agent_pair.agents[0]._state)

        dummy_pair = IndependentLearners([Random(5), Random(5)], args)
        agent_pair.agents[1], dummy_pair.agents[1] = (
            dummy_pair.agents[1],
            agent_pair.agents[1],
        )
        runner.train_loop(train_env, agent_pair, 1000, watchers)

        print("WarmUps Complete")
        agent_pair.agents[0]._state = agent_pair.agents[0]._state._replace(
            opt_state=agent1_fresh.opt_state,
            random_key=agent1_fresh.random_key,
        )
        agent_pair.agents[1] = dummy_pair.agents[1]

        if args.evo:
            num_iters = args.num_generations  # number of generations
        else:
            num_iters = int(
                args.total_timesteps / args.num_steps
            )  # number of episodes
        runner.train_loop(train_env, agent_pair, num_iters, watchers)


if __name__ == "__main__":
    main()
