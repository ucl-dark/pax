from datetime import datetime
from dbm import dumb
import os

# Added to test multiple devices
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import time
from typing import List, NamedTuple

from dm_env import TimeStep
from evosax import FitnessShaper
import hydra
import jax
import jax.numpy as jnp
import wandb
from pax.utils import save

# TODO: import when evosax library is updated
# from evosax.utils import ESLog
from pax.watchers import ESLog, cg_visitation, ipd_visitation

MAX_WANDB_CALLS = 1000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class EvoRunnerPMAP:
    """Holds the runner's state."""

    def __init__(self, args, strategy, es_params, param_reshaper, save_dir):
        self.algo = args.es.algo
        self.args = args
        self.es_params = es_params
        self.generations = 0
        self.num_opps = args.num_opps
        self.eval_steps = 0
        self.eval_episodes = 0
        self.param_reshaper = param_reshaper
        self.popsize = args.popsize
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_datetime = datetime.now()
        self.save_dir = save_dir
        self.start_time = time.time()
        self.strategy = strategy
        self.top_k = args.top_k
        self.train_steps = 0
        self.train_episodes = 0
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)
        self.num_devices = args.num_devices

        def pmap_reshape(cg_state):
            return jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (self.num_devices * self.popsize,) + x.shape[2:]
                ),
                cg_state,
            )

        def pmap_traj_reshape(traj):
            def fun(x):
                # (num_devices, num_trials, num_inner_steps, popsize, ... )
                # swapaxes
                # (num_devices, popsize, num_inner_steps, num_trials, ... )
                # reshape
                # (num_devices* popsize, num_inner_steps, num_trials ... )
                # swapaxes
                # (num_devices* popsize, num_trials, num_inner_steps ... )
                x = jnp.swapaxes(x, 1, 3)
                x = x.reshape((self.popsize * self.num_devices,) + x.shape[2:])
                x = jnp.swapaxes(x, 1, 2)
                return x

            return jax.tree_util.tree_map(fun, traj)

        self.pmap_reshape = pmap_reshape
        self.pmap_traj_reshape = pmap_traj_reshape

    def train_loop(self, env, agents, num_generations, watchers):
        """Run training of agents in environment"""

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                t1.observation,
                a1_mem,
            )

            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                t2.observation,
                a2_mem,
            )

            (tprime_1, tprime_2), env_state = env.batch_step(
                (a1, a2),
                env_state,
            )

            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                tprime_1.last(),
                a1_mem.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                tprime_2.last(),
                a2_mem.hidden,
            )
            return (
                tprime_1,
                tprime_2,
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=env.inner_episode_length,
            )

            # update second agent
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals

            # do second agent update
            final_t2 = t2._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], final_t2, a2_state, a2_mem
            )

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
            ), (*trajectories, a2_metrics)

        print("Training")
        print("------------------------------")
        log_interval = max(num_generations / MAX_WANDB_CALLS, 5)
        print(f"Number of Generations: {num_generations}")
        print(f"Number of Meta Episodes: {num_generations}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {env.num_envs}")
        print(f"Number of Opponent: {self.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        # Initialize evolution
        num_devices = self.num_devices
        num_gens = num_generations
        strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper
        popsize = self.popsize
        num_opps = self.num_opps
        evo_state = strategy.initialize(rng, es_params)
        fit_shaper = FitnessShaper(maximize=True)
        es_logging = ESLog(
            param_reshaper.total_params,
            num_gens,
            top_k=self.top_k,
            maximize=True,
        )
        log = es_logging.initialize()

        # Evolution specific: add pop size dimension
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step),
        )

        if self.args.env_type == "coin_game":
            env.batch_reset = jax.jit(jax.vmap(env.batch_reset))

        # these are batched
        pmap_init_hidden = jnp.tile(
            agent1._mem.hidden,
            (num_devices * popsize, num_opps, 1, 1),
        )

        dumb_rng = agent1._state.random_key
        vmap_state, _ = agent1.batch_init(
            jax.random.split(dumb_rng, num_devices * popsize),
            pmap_init_hidden,
        )

        # these are not batched by device number.
        init_hidden = jnp.tile(
            agent1._mem.hidden,
            (popsize, num_opps, 1, 1),
        )

        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(dumb_rng, popsize),
            init_hidden,
        )

        agent1._state = agent1._state._replace(params=vmap_state.params)
        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        def evo_rollout(params: jnp.ndarray, rng_device: jnp.ndarray):
            rng, rng_run, rng_gen, rng_key = jax.random.split(rng_device, 4)
            # this is serialized so pulls out initial state
            a1_state, a1_mem = agent1._state, agent1._mem

            # TODO: owe timon a drink if this is needed.
            # agent1._state, agent1._mem = agent1.batch_init(
            # jax.random.split(agent1._state.random_key, popsize),
            # init_hidden,
            # )

            # but who cares! i'm gonna replace this
            a1_state = a1_state._replace(
                params=params,
            )
            a1_mem = agent1.batch_reset(a1_mem, False)

            # init agent 2
            a2_state, a2_mem = agent2._state, agent2._mem
            a2_state, a2_mem = agent2.batch_init(
                jax.random.split(rng_key, popsize * num_opps).reshape(
                    self.popsize, num_opps, -1
                ),
                a2_mem.hidden,
            )

            # run the training loop!
            t_init, env_state = env.runner_reset(
                (popsize, num_opps, env.num_envs), rng_run
            )

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            traj_1, traj_2, a2_metrics = stack
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals

            # Fitness
            fitness = traj_1.rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = traj_2.rewards.mean(axis=(0, 1, 3, 4))
            return (
                fitness,
                other_fitness,
                env_state,
                traj_1,
                traj_2,
                a2_metrics,
            )

        pmap_rollout = jax.pmap(evo_rollout)

        for gen in range(num_gens):
            rng_evo, rng_devices = jax.random.split(rng, 2)
            rng_devices = jnp.tile(rng_devices, num_devices).reshape(
                num_devices, -1
            )
            # Ask
            x, evo_state = strategy.ask(
                rng_evo, evo_state, es_params
            )  # this means that x isn't of shape (total_popsize, params)
            a1_params = param_reshaper.reshape(
                x
            )  # this will reshape x into (num_devices, popsize, ....)
            # split x into chunks based on number devices
            (
                fitness,
                other_fitness,
                env_state,
                traj_1,
                traj_2,
                a2_metrics,
            ) = pmap_rollout(a1_params, rng_devices)
            fitness = jnp.reshape(fitness, popsize * num_devices)
            fitness_re = fit_shaper.apply(x, fitness)  # Maximize fitness

            env_state = self.pmap_reshape(env_state)
            traj_1, traj_2 = self.pmap_traj_reshape(
                traj_1
            ), self.pmap_traj_reshape(traj_2)
            # Tell
            evo_state = strategy.tell(
                x, fitness_re - fitness_re.mean(), evo_state, es_params
            )

            # Logging
            log = es_logging.update(log, x, fitness)
            if log["gen_counter"] % 100 == 0:
                log_savepath = os.path.join(
                    self.save_dir, f"generation_{log['gen_counter']}"
                )
                top_params = param_reshaper.reshape(log["top_gen_params"][0:1])
                # remove batch dimension
                top_params = jax.tree_util.tree_map(
                    lambda x: x.reshape(x.shape[1:]), top_params
                )
                save(top_params, log_savepath)

            if gen % log_interval == 0:
                if self.args.env_type == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(), self.cg_stats(env_state)
                    )
                    rewards_0 = traj_1.rewards.sum(axis=1).mean()
                    rewards_1 = traj_2.rewards.sum(axis=1).mean()

                elif self.args.env_type in [
                    "meta",
                    "sequential",
                ]:
                    final_t1 = t1._replace(
                        step_type=2 * jnp.ones_like(t1.step_type)
                    )
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(), self.ipd_stats(traj_1, final_t1)
                    )
                    rewards_0 = traj_1.rewards.mean()
                    rewards_1 = traj_2.rewards.mean()
                else:
                    env_stats = {}

                print(f"Generation: {gen}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"Fitness: {fitness.mean()} | Other Fitness: {other_fitness.mean()}"
                )
                print(
                    f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
                )
                print(f"Env Stats: {env_stats}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"Top 5: Generation | Mean: {log['log_top_gen_mean'][gen]}"
                    f" | Std: {log['log_top_gen_std'][gen]}"
                )
                print(
                    "--------------------------------------------------------------------------"
                )
                print(f"Agent {1} | Fitness: {log['top_gen_fitness'][0]}")
                print(f"Agent {2} | Fitness: {log['top_gen_fitness'][1]}")
                print(f"Agent {3} | Fitness: {log['top_gen_fitness'][2]}")
                print(f"Agent {4} | Fitness: {log['top_gen_fitness'][3]}")
                print(f"Agent {5} | Fitness: {log['top_gen_fitness'][4]}")
                print()

            self.generations += 1

            if watchers:
                wandb_log = {
                    "generations": self.generations,
                    "train/fitness/player_1": float(fitness.mean()),
                    "train/fitness/player_2": float(other_fitness.mean()),
                    "train/fitness/top_overall_mean": log["log_top_mean"][gen],
                    "train/fitness/top_overall_std": log["log_top_std"][gen],
                    "train/fitness/top_gen_mean": log["log_top_gen_mean"][gen],
                    "train/fitness/top_gen_std": log["log_top_gen_std"][gen],
                    "train/fitness/gen_std": log["log_gen_std"][gen],
                    "train/time/minutes": float(
                        (time.time() - self.start_time) / 60
                    ),
                    "train/time/seconds": float(
                        (time.time() - self.start_time)
                    ),
                    "train/episode_reward/player_1": float(rewards_0.mean()),
                    "train/episode_reward/player_2": float(rewards_1.mean()),
                }
                wandb_log = wandb_log | env_stats
                # loop through population
                for idx, (overall_fitness, gen_fitness) in enumerate(
                    zip(log["top_fitness"], log["top_gen_fitness"])
                ):
                    wandb_log[
                        f"train/fitness/top_overall_agent_{idx+1}"
                    ] = overall_fitness
                    wandb_log[
                        f"train/fitness/top_gen_agent_{idx+1}"
                    ] = gen_fitness

                # player 2 metrics
                # metrics [outer_timesteps, num_opps]
                flattened_metrics = jax.tree_util.tree_map(
                    lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                )
                agent2._logger.metrics = (
                    agent2._logger.metrics | flattened_metrics
                )

                agent1._logger.metrics = (
                    agent1._logger.metrics | flattened_metrics
                )
                agents.log(watchers)
                wandb.log(wandb_log)

        return agents


#### Copy of experiment.py so we can just run this file for debugging ####

from datetime import datetime
import logging
import os

# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

from evosax import OpenES, CMA_ES, PGPE, ParameterReshaper, SimpleGA
import hydra
import omegaconf
import wandb

from pax.env_inner import SequentialMatrixGame
from pax.hyper.ppo import make_hyper
from pax.learners import IndependentLearners, EvolutionaryLearners
from pax.env_inner import InfiniteMatrixGame
from pax.env_meta import CoinGame, MetaFiniteGame
from pax.naive.naive import make_naive_pg
from pax.naive_exact import NaiveExact
from pax.ppo.ppo import make_agent
from pax.ppo.ppo_gru import make_gru_agent
from pax.runner_rl import Runner
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
    save_dir = f"{args.save_dir}/{str(datetime.now()).replace(' ', '_')}"
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

        return EvoRunner(args, strategy, es_params, param_reshaper, save_dir)
    else:
        return Runner(args)


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

    # flake8: noqa: C901
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

    strategies = {
        "TitForTat": TitForTat,
        "Defect": Defect,
        "Altruistic": Altruistic,
        "Human": Human,
        "Random": get_random_agent,
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

    if args.evo:
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
        save_dir = global_setup(args)

    with Section("Env setup", logger=logger):
        train_env, test_env = env_setup(args, logger)

    with Section("Agent setup", logger=logger):
        agent_pair = agent_setup(args, logger)

    with Section("Watcher setup", logger=logger):
        watchers = watcher_setup(args, logger)

    with Section("Runner setup", logger=logger):
        runner = runner_setup(args, agent_pair, save_dir, logger)

    num_generations = args.num_generations
    if not args.wandb.log:
        watchers = False
    runner.train_loop(train_env, agent_pair, num_generations, watchers)


if __name__ == "__main__":
    main()
