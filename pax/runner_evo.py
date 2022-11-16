import os
import time
from datetime import datetime
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from evosax import FitnessShaper

import wandb
from pax.utils import MemoryState, TrainingState, save

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


class EvoRunner:
    """
    Evoluationary Strategy runner provides a convenient example for quickly writing
    a MARL runner for PAX. The EvoRunner class can be used to
    run an RL agent (optimised by an Evolutionary Strategy) against an Reinforcement Learner.
    It composes together agents, watchers, and the environment.
    Within the init, we declare vmaps and pmaps for training.
    The environment provided must conform to a meta-environment.
    Args:
        agents (Tuple[agents]):
            The set of agents that will run in the experiment. Note, ordering is
             important for logic used in the class.
        env (gymnax.envs.Environment):
            The meta-environment that the agents will run in.
        strategy (evosax.Strategy):
            The evolutionary strategy that will be used to train the agents.
        param_reshaper (evosax.param_reshaper.ParameterReshaper):
            A function that reshapes the parameters of the agents into a format that can be
             used by the strategy.
        save_dir (string):
            The directory to save the model to.
        args (NamedTuple):
            A tuple of experiment arguments used (usually provided by HydraConfig).
    """

    def __init__(
        self, agents, env, strategy, es_params, param_reshaper, save_dir, args
    ):
        self.args = args
        self.algo = args.es.algo
        self.es_params = es_params
        self.generations = 0
        self.num_opps = args.num_opps
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
        self.cg_stats = jax.jit(jax.vmap(cg_visitation))

        # Evo Runner has 3 vmap dims (popsize, num_opps, num_envs)
        # Evo Runner also has an additional pmap dim (num_devices, ...)
        # For the env we vmap over the rng but not params

        # num envs
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )

        # num opps
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )
        # pop size
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )
        self.split = jax.vmap(
            jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None)),
            (0, None),
        )

        num_outer_steps = (
            1
            if self.args.env_type == "sequential"
            else self.args.num_steps // self.args.num_inner_steps
        )

        agent1, agent2 = agents

        # vmap agents accordingly
        # agent 1 is batched over popsize and num_opps
        agent1.batch_init = jax.vmap(
            jax.vmap(
                agent1.make_initial_state,
                (None, 0),  # (params, rng)
                (None, 0),  # (TrainingState, MemoryState)
            ),
            # both for Population
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent1.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent1.batch_policy = jax.jit(
            jax.vmap(
                jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0)),
            )
        )

        if args.agent2 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent2.batch_init = jax.jit(
                jax.vmap(jax.vmap(agent2.make_initial_state))
            )
        else:
            agent2.batch_init = jax.jit(
                jax.vmap(
                    jax.vmap(agent2.make_initial_state, (0, None), 0),
                    (0, None),
                    0,
                )
            )

        agent2.batch_policy = jax.jit(jax.vmap(jax.vmap(agent2._policy, 0, 0)))
        agent2.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent2.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent2.batch_update = jax.jit(
            jax.vmap(
                jax.vmap(agent2.update, (1, 0, 0, 0)),
                (1, 0, 0, 0),
            )
        )
        if args.agent2 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))

            key = jax.random.split(
                agent2._state.random_key, args.popsize * args.num_opps
            ).reshape(args.popsize, args.num_opps, -1)

            agent2._state, agent2._mem = agent2.batch_init(
                key,
                init_hidden,
            )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, :, 0, :]
            # a1_rng = rngs[:, :, :, 1, :]
            # a2_rng = rngs[:, :, :, 2, :]
            rngs = rngs[:, :, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                obs1,
                a1_mem,
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                obs2,
                a2_mem,
            )
            (next_obs1, next_obs2), env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                (a1, a2),
                env_params,
            )

            traj1 = Sample(
                obs1,
                a1,
                rewards[0],
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )
            traj2 = Sample(
                obs2,
                a2,
                rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
            )
            return (
                rngs,
                next_obs1,
                next_obs2,
                rewards[0],
                rewards[1],
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
                env_params,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=args.num_inner_steps,
            )
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                a1_mem = agent1.meta_policy(a1_mem)

            # update second agent
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1],
                obs2,
                a2_state,
                a2_mem,
            )
            return (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ), (*trajectories, a2_metrics)

        def _rollout(
            _params: jnp.ndarray,
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
                * args.num_opps
                * args.popsize
            ).reshape((args.popsize, args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.popsize, args.num_opps, args.num_envs)),
                jnp.zeros((args.popsize, args.num_opps, args.num_envs)),
            ]

            # Player 1
            _a1_state = _a1_state._replace(params=_params)
            _a1_mem = agent1.batch_reset(_a1_mem, False)
            # Player 2
            if args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(obs[1])

            else:
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(
                        _rng_run, args.popsize * args.num_opps
                    ).reshape(args.popsize, args.num_opps, -1),
                    agent2._mem.hidden,
                )

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    _a1_state,
                    _a1_mem,
                    a2_state,
                    a2_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=num_outer_steps,
            )

            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                _a1_state,
                _a1_mem,
                a2_state,
                a2_mem,
                env_state,
                _env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # Fitness
            fitness = traj_1.rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = traj_2.rewards.mean(axis=(0, 1, 3, 4))
            # Stats
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x,
                    self.cg_stats(env_state),
                )

                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()

            elif args.env_id in [
                "matrix_game",
            ]:
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj_1.observations,
                        traj_1.actions,
                        obs1,
                    ),
                )
                rewards_1 = traj_1.rewards.mean()
                rewards_2 = traj_2.rewards.mean()
            return (
                fitness,
                other_fitness,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
            )

        self.rollout = jax.pmap(
            _rollout,
            in_axes=(0, None, None, None, None),
        )

    def run_loop(
        self,
        env_params,
        agents,
        num_generations: int,
        watchers: Callable,
    ):
        """Run training of agents in environment"""
        print("Training")
        print("------------------------------")
        log_interval = max(num_generations / MAX_WANDB_CALLS, 5)
        print(f"Number of Generations: {num_generations}")
        print(f"Number of Meta Episodes: {num_generations}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {self.args.num_envs}")
        print(f"Number of Opponent: {self.args.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        # Initialize evolution
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
        num_devices = self.args.num_devices

        # Reshape a single agent's params before vmapping
        init_hidden = jnp.tile(
            agent1._mem.hidden,
            (popsize, num_opps, 1, 1),
        )
        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(agent1._state.random_key, popsize),
            init_hidden,
        )

        a1_state, a1_mem = agent1._state, agent1._mem

        for gen in range(num_gens):
            rng, rng_run, rng_gen, rng_key = jax.random.split(rng, 4)

            # Ask
            x, evo_state = strategy.ask(rng_gen, evo_state, es_params)
            params = param_reshaper.reshape(x)
            if num_devices == 1:
                params = jax.tree_util.tree_map(
                    lambda x: jax.lax.expand_dims(x, (0,)), params
                )
            # Evo Rollout
            (
                fitness,
                other_fitness,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
            ) = self.rollout(params, rng_run, a1_state, a1_mem, env_params)

            # Reshape over devices
            fitness = jnp.reshape(fitness, popsize * num_devices)
            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), env_stats)

            # Maximize fitness
            fitness_re = fit_shaper.apply(x, fitness)

            # Tell
            evo_state = strategy.tell(
                x, fitness_re - fitness_re.mean(), evo_state, es_params
            )
            # Logging
            log = es_logging.update(log, x, fitness)

            # Saving
            if self.args.save and gen % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"generation_{gen}")
                if num_devices > 1:
                    top_params = param_reshaper.reshape(
                        log["top_gen_params"][0 : self.args.num_devices]
                    )
                    top_params = jax.tree_util.tree_map(
                        lambda x: x[0].reshape(x[0].shape[1:]), top_params
                    )
                else:
                    top_params = param_reshaper.reshape(
                        log["top_gen_params"][0:1]
                    )
                    top_params = jax.tree_util.tree_map(
                        lambda x: x.reshape(x.shape[1:]), top_params
                    )
                save(top_params, log_savepath)
                if watchers:
                    print(f"Saving generation {gen} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {gen} locally")

            if gen % log_interval == 0:
                print(f"Generation: {gen}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"Fitness: {fitness.mean()} | Other Fitness: {other_fitness.mean()}"
                )
                print(
                    f"Total Episode Reward: {float(rewards_1.mean()), float(rewards_2.mean())}"
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

            if watchers:
                wandb_log = {
                    "generations": gen,
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
                    "train/episode_reward/player_1": float(rewards_1.mean()),
                    "train/episode_reward/player_2": float(rewards_2.mean()),
                }
                wandb_log.update(env_stats)
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

                agent2._logger.metrics.update(flattened_metrics)
                for watcher, agent in zip(watchers, agents):
                    watcher(agent)
                wandb.log(wandb_log)

        return agents
