import logging
import os
from datetime import datetime
from functools import partial

import gymnax
import hydra
import jax
import jax.numpy as jnp
import omegaconf
from evosax import CMA_ES, PGPE, OpenES, ParameterReshaper, SimpleGA

import wandb
from pax.agents.hyper.ppo import make_hyper
from pax.agents.lola.lola import make_lola
from pax.agents.mfos_ppo.ppo_gru import make_mfos_agent
from pax.agents.naive.naive import make_naive_pg
from pax.agents.naive_exact import NaiveExact
from pax.agents.ppo.ppo import make_agent
from pax.agents.ppo.ppo_gru import make_gru_agent
from pax.agents.strategies import (
    Altruistic,
    Defect,
    EvilGreedy,
    GoodGreedy,
    GrimTrigger,
    HyperAltruistic,
    HyperDefect,
    HyperTFT,
    Random,
    RandomGreedy,
    Stay,
    TitForTat,
)
from pax.agents.tensor_strategies import (
    TitForTatCooperate,
    TitForTatDefect,
    TitForTatHarsh,
    TitForTatSoft,
    TitForTatStrictStay,
    TitForTatStrictSwitch,
)
from pax.envs.coin_game import CoinGame
from pax.envs.coin_game import EnvParams as CoinGameParams
from pax.envs.in_the_matrix import EnvParams as InTheMatrixParams
from pax.envs.in_the_matrix import InTheMatrix
from pax.envs.infinite_matrix_game import EnvParams as InfiniteMatrixGameParams
from pax.envs.infinite_matrix_game import InfiniteMatrixGame
from pax.envs.iterated_matrix_game import EnvParams as IteratedMatrixGameParams
from pax.envs.iterated_matrix_game import IteratedMatrixGame
from pax.envs.iterated_tensor_game_n_player import (
    EnvParams as IteratedTensorGameNPlayerParams,
)
from pax.envs.iterated_tensor_game_n_player import IteratedTensorGameNPlayer
from pax.runners.runner_eval import EvalRunner
from pax.runners.runner_eval_multishaper import MultishaperEvalRunner
from pax.runners.runner_evo import EvoRunner
from pax.runners.runner_evo_multishaper import MultishaperEvoRunner
from pax.runners.runner_ipditm_eval import IPDITMEvalRunner
from pax.runners.runner_marl import RLRunner
from pax.runners.runner_marl_nplayer import NplayerRLRunner
from pax.runners.runner_sarl import SARLRunner
from pax.utils import Section
from pax.watchers import (
    logger_hyper,
    logger_naive_exact,
    losses_naive,
    losses_ppo,
    naive_pg_losses,
    policy_logger_ppo,
    policy_logger_ppo_with_memory,
    value_logger_ppo,
)

# NOTE: THIS MUST BE sDONE BEFORE IMPORTING JAX
# uncomment to debug multi-devices on CPU
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
# from jax.config import config

# config.update("jax_disable_jit", True)


def global_setup(args):
    """Set up global variables."""
    save_dir = f"{args.save_dir}/{str(datetime.now()).replace(' ', '_').replace(':', '.')}"
    if not args.runner == "eval":
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
            ),  # type: ignore
            settings=wandb.Settings(code_dir="."),
        )
        wandb.run.log_code(".")
    return save_dir


def env_setup(args, logger=None):
    """Set up env variables."""

    if args.env_id == "iterated_matrix_game":
        payoff = jnp.array(args.payoff)

        env = IteratedMatrixGame(
            num_inner_steps=args.num_inner_steps,
            num_outer_steps=args.num_outer_steps,
        )
        env_params = IteratedMatrixGameParams(payoff_matrix=payoff)
        if logger:
            logger.info(
                f"Env Type: {args.env_type} | Inner Episode Length: {args.num_inner_steps}"
            )
            logger.info(f"Outer Episode Length: {args.num_outer_steps}")
    elif args.env_id == "iterated_nplayer_tensor_game":
        payoff = jnp.array(args.payoff_table)

        env = IteratedTensorGameNPlayer(
            num_players=args.num_players,
            num_inner_steps=args.num_inner_steps,
            num_outer_steps=args.num_outer_steps,
        )
        env_params = IteratedTensorGameNPlayerParams(payoff_table=payoff)

        if logger:
            logger.info(
                f"Env Type: {args.env_type} s| Inner Episode Length: {args.num_inner_steps}"
            )
    elif args.env_id == "infinite_matrix_game":
        payoff = jnp.array(args.payoff)
        env = InfiniteMatrixGame(num_steps=args.num_steps)
        env_params = InfiniteMatrixGameParams(
            payoff_matrix=payoff, gamma=args.ppo.gamma
        )
        if logger:
            logger.info(
                f"Game Type: Infinite | Inner Discount: {args.env_discount}"
            )

    elif args.env_id == "coin_game":
        payoff = jnp.array(args.payoff)
        env_params = CoinGameParams(payoff_matrix=payoff)
        if args.ppo1.with_cnn != args.ppo2.with_cnn:
            raise ValueError(
                "Both agents must have the same CNN configuration."
            )

        env = CoinGame(
            num_inner_steps=args.num_inner_steps,
            num_outer_steps=args.num_outer_steps,
            cnn=args.ppo1.with_cnn,
            egocentric=args.egocentric,
        )
        if logger:
            logger.info(
                f"Env Type: CoinGame | Inner Episode Length: {args.num_inner_steps}"
            )
            logger.info(f"Outer Episode Length: {args.num_outer_steps}")

    elif args.env_id == "InTheMatrix":
        payoff = jnp.array(args.payoff)
        env_params = InTheMatrixParams(
            payoff_matrix=payoff, freeze_penalty=args.freeze
        )
        env = InTheMatrix(
            num_inner_steps=args.num_inner_steps,
            num_outer_steps=args.num_outer_steps,
            fixed_coin_location=args.fixed_coins,
        )
        if logger:
            logger.info(
                f"Env Type: InTheMatrix | Inner Episode Length: {args.num_inner_steps}"
            )
    elif args.runner == "sarl":
        env, env_params = gymnax.make(args.env_id)
    else:
        raise ValueError(f"Unknown env id {args.env_id}")
    return env, env_params


def runner_setup(args, env, agents, save_dir, logger):
    if args.runner == "eval":
        logger.info("Evaluating with EvalRunner")
        return EvalRunner(agents, env, args)
    elif args.runner == "multishaper_eval":
        logger.info("Training with multishaper eval Runner")
        return MultishaperEvalRunner(agents, env, save_dir, args)
    elif args.runner == "ipditm_eval":
        logger.info("Evaluating with ipditmEvalRunner")
        return IPDITMEvalRunner(agents, env, save_dir, args)

    if args.runner == "evo" or args.runner == "multishaper_evo":
        agent1 = agents[0]
        algo = args.es.algo
        strategies = {"CMA_ES", "OpenES", "PGPE", "SimpleGA"}
        assert algo in strategies, f"{algo} not in evolution strategies"

        def get_ga_strategy(agent):
            """Returns the SimpleGA strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(
                agent._state.params, n_devices=args.num_devices
            )
            strategy = SimpleGA(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize,
            )
            es_params = strategy.default_params
            return strategy, es_params, param_reshaper

        def get_cma_strategy(agent):
            """Returns the CMA strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(
                agent._state.params, n_devices=args.num_devices
            )
            strategy = CMA_ES(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize,
                elite_ratio=args.es.elite_ratio,
            )
            es_params = strategy.default_params
            return strategy, es_params, param_reshaper

        def get_openes_strategy(agent):
            """Returns the OpenES strategy, es params, and param_reshaper"""
            param_reshaper = ParameterReshaper(
                agent._state.params, n_devices=args.num_devices
            )
            strategy = OpenES(
                num_dims=param_reshaper.total_params,
                popsize=args.popsize * args.num_devices,
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
        if args.runner == "evo":
            logger.info("Training with EVO runner")
            return EvoRunner(
                agents,
                env,
                strategy,
                es_params,
                param_reshaper,
                save_dir,
                args,
            )
        elif args.runner == "multishaper_evo":
            logger.info("Training with multishaper EVO runner")
            return MultishaperEvoRunner(
                agents,
                env,
                strategy,
                es_params,
                param_reshaper,
                save_dir,
                args,
            )

    elif args.runner == "rl":
        logger.info("Training with RL Runner")
        return RLRunner(agents, env, save_dir, args)
    elif args.runner == "tensor_rl_nplayer":
        return NplayerRLRunner(agents, env, save_dir, args)
    elif args.runner == "sarl":
        logger.info("Training with SARL Runner")
        return SARLRunner(agents, env, save_dir, args)
    else:
        raise ValueError(f"Unknown runner type {args.runner}")


# flake8: noqa: C901
def agent_setup(args, env, env_params, logger):
    """Set up agent variables."""
    if (
        args.env_id == "iterated_matrix_game"
        or args.env_id == "iterated_nplayer_tensor_game"
    ):
        obs_shape = env.observation_space(env_params).n
    elif args.env_id == "InTheMatrix":
        obs_shape = jax.tree_map(
            lambda x: x.shape, env.observation_space(env_params)
        )
    else:
        obs_shape = env.observation_space(env_params).shape

    num_actions = env.num_actions

    def get_LOLA_agent(seed, player_id):
        return make_lola(
            args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            seed=seed,
            player_id=player_id,
            env_params=env_params,
            env_step=env.step,
            env_reset=env.reset,
        )

    def get_PPO_memory_agent(seed, player_id):
        player_args = omegaconf.OmegaConf.select(args, "ppo" + str(player_id))
        num_iterations = args.num_iters
        if player_id == 1 and args.env_type == "meta":
            num_iterations = args.num_outer_steps
        return make_gru_agent(
            args,
            player_args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            seed=seed,
            num_iterations=num_iterations,
            player_id=player_id,
        )

    def get_PPO_agent(seed, player_id):
        player_args = omegaconf.OmegaConf.select(args, "ppo" + str(player_id))

        if player_id == 1 and args.env_type == "meta":
            num_iterations = args.num_outer_steps
        else:
            num_iterations = args.num_iters
        ppo_agent = make_agent(
            args,
            player_args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            num_iterations=num_iterations,
            seed=seed,
            player_id=player_id,
        )
        return ppo_agent

    def get_PPO_tabular_agent(seed, player_id):
        player_args = args.ppo1 if player_id == 1 else args.ppo2
        num_iterations = args.num_iters
        if player_id == 1 and args.env_type == "meta":
            num_iterations = args.num_outer_steps
        ppo_agent = make_agent(
            args,
            player_args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            seed=seed,
            num_iterations=num_iterations,
            player_id=player_id,
            tabular=True,
        )
        return ppo_agent

    def get_mfos_agent(seed, player_id):
        agent_args = args.ppo1
        num_iterations = args.num_iters
        if player_id == 1 and args.env_type == "meta":
            num_iterations = args.num_outer_steps
        ppo_agent = make_mfos_agent(
            args,
            agent_args,
            obs_spec=obs_shape,
            num_iterations=num_iterations,
            action_spec=num_actions,
            seed=seed,
            player_id=player_id,
        )
        return ppo_agent

    def get_hyper_agent(seed, player_id):
        hyper_agent = make_hyper(
            args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            seed=seed,
            player_id=player_id,
        )
        return hyper_agent

    def get_naive_pg(seed, player_id):
        naive_agent = make_naive_pg(
            args,
            obs_spec=obs_shape,
            action_spec=num_actions,
            seed=seed,
            player_id=player_id,
        )
        return naive_agent

    def get_naive_learner(seed, player_id):
        agent = NaiveExact(
            action_dim=num_actions,
            env_params=env_params,
            lr=args.naive.lr,
            num_envs=args.num_envs,
            player_id=player_id,
        )
        return agent

    def get_random_agent(seed, player_id):
        random_agent = Random(num_actions, args.num_envs, obs_shape)
        random_agent.player_id = player_id
        return random_agent

    # flake8: noqa: C901
    def get_stay_agent(seed, player_id):
        agent = Stay(num_actions, args.num_envs, args.num_players)
        agent.player_id = player_id
        return agent

    strategies = {
        "TitForTat": partial(TitForTat, args.num_envs),
        "TitForTatStrictStay": partial(TitForTatStrictStay, args.num_envs),
        "TitForTatStrictSwitch": partial(TitForTatStrictSwitch, args.num_envs),
        "TitForTatCooperate": partial(TitForTatCooperate, args.num_envs),
        "TitForTatDefect": partial(TitForTatDefect, args.num_envs),
        "TitForTatHarsh": partial(TitForTatHarsh, args.num_envs),
        "TitForTatSoft": partial(TitForTatSoft, args.num_envs),
        "Defect": partial(Defect, args.num_envs),
        "Altruistic": partial(Altruistic, args.num_envs),
        "Random": get_random_agent,
        "Stay": get_stay_agent,
        "Grim": partial(GrimTrigger, args.num_envs),
        "GoodGreedy": partial(GoodGreedy, args.num_envs),
        "EvilGreedy": partial(EvilGreedy, args.num_envs),
        "RandomGreedy": partial(RandomGreedy, args.num_envs),
        "LOLA": get_LOLA_agent,
        "PPO": get_PPO_agent,
        "PPO_memory": get_PPO_memory_agent,
        "Naive": get_naive_pg,
        "Tabular": get_PPO_tabular_agent,
        "MFOS": get_mfos_agent,
        # HyperNetworks
        "Hyper": get_hyper_agent,
        "NaiveEx": get_naive_learner,
        "HyperAltruistic": partial(HyperAltruistic, args.num_envs),
        "HyperDefect": partial(HyperDefect, args.num_envs),
        "HyperTFT": partial(HyperTFT, args.num_envs),
    }

    if args.runner == "sarl":
        assert args.agent1 in strategies
        num_agents = 1
        seeds = [args.seed]
        # Create Player IDs by normalizing seeds to 1, 2 respectively
        pids = [0]
        agent_1 = strategies[args.agent1](seeds[0], pids[0])  # player 1

        if args.agent1 in ["PPO", "PPO_memory"] and args.ppo.with_cnn:
            logger.info(f"PPO with CNN: {args.ppo.with_cnn}")
        logger.info(f"Agent Pair: {args.agent1}")
        logger.info(f"Agent seeds: {seeds[0]}")

        if args.runner in ["eval", "sarl"]:
            logger.info("Using Independent Learners")
            return agent_1

    else:
        for i in range(1, args.num_players + 1):
            assert (
                omegaconf.OmegaConf.select(args, "agent" + str(i))
                in strategies
            )

        seeds = [
            seed for seed in range(args.seed, args.seed + args.num_players)
        ]
        # Create Player IDs by normalizing seeds to 1, 2, 3 ..n respectively
        pids = [
            seed % seed + i if seed != 0 else 1
            for seed, i in zip(seeds, range(1, args.num_players + 1))
        ]
        agents = []
        for i in range(args.num_players):
            agents.append(
                strategies[
                    omegaconf.OmegaConf.select(args, "agent" + str(i + 1))
                ](seeds[i], pids[i])
            )
        logger.info(
            f"Agent Pair: {[omegaconf.OmegaConf.select(args, 'agent' + str(i)) for i in range(1, args.num_players + 1)]}"
        )
        logger.info(f"Agent seeds: {seeds}")

        return agents


def watcher_setup(args, logger):
    """Set up watcher variables."""

    def ppo_memory_log(
        agent,
    ):
        losses = losses_ppo(agent)
        if args.env_id not in [
            "coin_game",
            "InTheMatrix",
            "iterated_matrix_game",
            "iterated_nplayer_tensor_game",
        ]:
            policy = policy_logger_ppo_with_memory(agent)
            losses.update(policy)
        return

    def ppo_log(agent):
        losses = losses_ppo(agent)
        if args.env_id not in [
            "coin_game",
            "InTheMatrix",
            "iterated_matrix_game",
            "iterated_nplayer_tensor_game",
        ]:
            policy = policy_logger_ppo(agent)
            value = value_logger_ppo(agent)
            losses.update(value)
            losses.update(policy)
        return

    def dumb_log(agent, *args):
        return

    def hyper_log(agent):
        losses = losses_ppo(agent)
        policy = logger_hyper(agent)
        losses.update(policy)
        if args.wandb.log:
            losses = jax.tree_util.tree_map(
                lambda x: x.item() if isinstance(x, jax.Array) else x, losses
            )
            wandb.log(losses)
        return

    def naive_logger(agent):
        losses = losses_naive(agent)
        policy = logger_naive_exact(agent)
        losses.update(policy)
        if args.wandb.log:
            wandb.log(losses)
        return

    def naive_pg_log(agent):
        losses = naive_pg_losses(agent)
        if args.env_id in [
            "iterated_matrix_game",
            "iterated_nplayer_tensor_game",
        ]:
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
        "GoodGreedy": dumb_log,
        "EvilGreedy": dumb_log,
        "RandomGreedy": dumb_log,
        "MFOS": dumb_log,
        "PPO": ppo_log,
        "LOLA": dumb_log,
        "PPO_memory": ppo_memory_log,
        "Naive": naive_pg_log,
        "Hyper": hyper_log,
        "NaiveEx": naive_logger,
        "HyperAltruistic": dumb_log,
        "HyperDefect": dumb_log,
        "HyperTFT": dumb_log,
        "Tabular": ppo_log,
        "PPO_memory_pretrained": ppo_memory_log,
        "MFOS_pretrained": dumb_log,
    }

    if args.runner == "sarl":
        assert args.agent1 in strategies

        agent_1_log = naive_pg_log  # strategies[args.agent1] #

        return agent_1_log
    else:
        agent_log = []
        for i in range(1, args.num_players + 1):
            assert getattr(args, f"agent{i}") in strategies
            agent_log.append(strategies[getattr(args, f"agent{i}")])
        return agent_log


@hydra.main(config_path="conf", config_name="config")
def main(args):
    """Set up main."""
    logger = logging.getLogger()
    with Section("Global setup", logger=logger):
        save_dir = global_setup(args)

    with Section("Env setup", logger=logger):
        env, env_params = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent_pair = agent_setup(args, env, env_params, logger)
    with Section("Watcher setup", logger=logger):
        watchers = watcher_setup(args, logger)

    with Section("Runner setup", logger=logger):
        runner = runner_setup(args, env, agent_pair, save_dir, logger)

    if not args.wandb.log:
        watchers = False

    print(f"Number of Training Iterations: {args.num_iters}")

    if args.runner == "evo" or args.runner == "multishaper_evo":
        runner.run_loop(env_params, agent_pair, args.num_iters, watchers)
    elif args.runner == "rl" or args.runner == "tensor_rl_nplayer":
        # number of episodes
        print(f"Number of Episodes: {args.num_iters}")
        runner.run_loop(env_params, agent_pair, args.num_iters, watchers)

    elif args.runner == "ipditm_eval" or args.runner == "multishaper_eval":
        runner.run_loop(env_params, agent_pair, watchers)

    elif args.runner == "sarl":
        print(f"Number of Episodes: {args.num_iters}")
        runner.run_loop(env, env_params, agent_pair, args.num_iters, watchers)
    elif args.runner == "eval":
        print(f"Number of Episodes: {args.num_iters}")
        runner.run_loop(env, env_params, agent_pair, args.num_iters, watchers)

    wandb.finish()


if __name__ == "__main__":
    main()
