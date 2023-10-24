import os
import time
from datetime import datetime
from typing import Any, Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import FitnessShaper

import wandb
from pax.utils import MemoryState, TrainingState, save, float_precision, Sample

# TODO: import when evosax library is updated
# from evosax.utils import ESLog
from pax.watchers import ESLog, cg_visitation, ipd_visitation, ipditm_stats
from pax.watchers.fishery import fishery_stats
from pax.watchers.cournot import cournot_stats
from pax.watchers.rice import rice_stats
from pax.watchers.c_rice import c_rice_stats

MAX_WANDB_CALLS = 1000


class EvoRunnerNRoles:
    """
    This Runner extends the EvoRunner class with three features:
    1. Allow for both the first and second agent to assume multiple roles in the game.
    2. Allow for shuffling of these roles for each rollout.
    3. Enable annealed self_play via the self_play_anneal flag.
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

    # TODO fix C901 (function too complex)
    # flake8: noqa: C901
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
        self.ipditm_stats = jax.jit(
            jax.vmap(ipditm_stats, in_axes=(0, 2, 2, None))
        )

        if args.num_players != args.agent1_roles + args.agent2_roles:
            raise ValueError("Number of players must match number of roles")

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

        self.num_outer_steps = args.num_outer_steps
        agent1, agent2 = agents[0], agents[1]

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
            jax.vmap(jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))),
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
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))

            a2_rng = jnp.concatenate(
                [jax.random.split(agent2._state.random_key, args.num_opps)]
                * args.popsize
            ).reshape(args.popsize, args.num_opps, -1)

            agent2._state, agent2._mem = agent2.batch_init(
                a2_rng,
                init_hidden,
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

        # jit evo
        strategy.ask = jax.jit(strategy.ask)
        strategy.tell = jax.jit(strategy.tell)
        param_reshaper.reshape = jax.jit(param_reshaper.reshape)

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
                agent_order,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, :, 0, :]
            rngs = rngs[:, :, :, 3, :]

            a1_actions = []
            new_a1_memories = []
            for _obs, _mem in zip(obs1, a1_mem):
                a1_action, a1_state, new_a1_memory = agent1.batch_policy(
                    a1_state,
                    _obs,
                    _mem,
                )
                a1_actions.append(a1_action)
                new_a1_memories.append(new_a1_memory)

            a2_actions = []
            new_a2_memories = []
            for _obs, _mem in zip(obs2, a2_mem):
                a2_action, a2_state, new_a2_memory = agent2.batch_policy(
                    a2_state,
                    _obs,
                    _mem,
                )
                a2_actions.append(a2_action)
                new_a2_memories.append(new_a2_memory)

            actions = jnp.asarray([*a1_actions, *a2_actions])[agent_order]
            obs, env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                tuple(actions),
                env_params,
            )

            inv_agent_order = jnp.argsort(agent_order)
            obs = jnp.asarray(obs)[inv_agent_order]
            rewards = jnp.asarray(rewards)[inv_agent_order]
            agent1_roles = len(a1_actions)

            a1_trajectories = [
                Sample(
                    observation,
                    action,
                    reward * jnp.logical_not(done),
                    new_memory.extras["log_probs"],
                    new_memory.extras["values"],
                    done,
                    memory.hidden,
                )
                for observation, action, reward, new_memory, memory in zip(
                    obs1,
                    a1_actions,
                    rewards[:agent1_roles],
                    new_a1_memories,
                    a1_mem,
                )
            ]
            a2_trajectories = [
                Sample(
                    observation,
                    action,
                    reward * jnp.logical_not(done),
                    new_memory.extras["log_probs"],
                    new_memory.extras["values"],
                    done,
                    memory.hidden,
                )
                for observation, action, reward, new_memory, memory in zip(
                    obs2,
                    a2_actions,
                    rewards[agent1_roles:],
                    new_a2_memories,
                    a2_mem,
                )
            ]

            return (
                rngs,
                tuple(obs[:agent1_roles]),
                tuple(obs[agent1_roles:]),
                tuple(rewards[:agent1_roles]),
                tuple(rewards[agent1_roles:]),
                a1_state,
                tuple(new_a1_memories),
                a2_state,
                tuple(new_a2_memories),
                env_state,
                env_params,
                agent_order,
            ), (
                a1_trajectories,
                a2_trajectories,
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
                agent_order,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                a1_mem = [agent1.meta_policy(_a1_mem) for _a1_mem in a1_mem]

            # update second agent
            new_a2_memories = []
            a2_metrics = None
            for _obs, mem, traj in zip(obs2, a2_mem, trajectories[1]):
                a2_state, a2_mem, a2_metrics = agent2.batch_update(
                    traj,
                    _obs,
                    a2_state,
                    mem,
                )
                new_a2_memories.append(a2_mem)
            return (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                tuple(new_a2_memories),
                env_state,
                env_params,
                agent_order,
            ), (*trajectories, a2_metrics)

        def _rollout(
            _params: jnp.ndarray,
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _env_params: Any,
            roles: Tuple[int, int],
        ):
            # env reset
            env_rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
                * args.num_opps
                * args.popsize
            ).reshape((args.popsize, args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(env_rngs, _env_params)
            rewards = [
                jnp.zeros(
                    (args.popsize, args.num_opps, args.num_envs),
                    dtype=float_precision,
                )
            ] * (1 + args.agent2_roles)

            # Player 1
            _a1_state = _a1_state._replace(params=_params)
            _a1_mem = agent1.batch_reset(_a1_mem, False)
            # Player 2
            if args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(obs[1])
            else:
                # meta-experiments - init 2nd agent per trial
                a2_rng = jnp.concatenate(
                    [jax.random.split(_rng_run, args.num_opps)] * args.popsize
                ).reshape(args.popsize, args.num_opps, -1)
                a2_state, a2_mem = agent2.batch_init(
                    a2_rng,
                    agent2._mem.hidden,
                )

            if _a2_state is not None:
                a2_state = _a2_state

            agent_order = jnp.arange(args.num_players)
            if args.shuffle_players:
                agent_order = jax.random.permutation(_rng_run, agent_order)

            agent1_roles, agent2_roles = roles
            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    env_rngs,
                    # Split obs and rewards between agents
                    tuple(obs[:agent1_roles]),
                    tuple(obs[agent1_roles:]),
                    tuple(rewards[:agent1_roles]),
                    tuple(rewards[agent1_roles:]),
                    _a1_state,
                    (_a1_mem,) * agent1_roles,
                    a2_state,
                    (a2_mem,) * agent2_roles,
                    env_state,
                    _env_params,
                    agent_order,
                ),
                None,
                length=self.num_outer_steps,
            )

            (
                env_rngs,
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
                agent_order,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # Fitness
            agent_1_rewards = jnp.concatenate(
                [traj.rewards for traj in traj_1]
            )
            fitness = agent_1_rewards.mean(axis=(0, 1, 3, 4))
            # At the end of self play annealing there will be no agent2 reward
            if agent2_roles > 0:
                agent_2_rewards = jnp.concatenate(
                    [traj.rewards for traj in traj_2]
                )
            else:
                agent_2_rewards = jnp.zeros_like(agent_1_rewards)
            other_fitness = agent_2_rewards.mean(axis=(0, 1, 3, 4))
            rewards_1 = agent_1_rewards.mean()
            rewards_2 = agent_2_rewards.mean()

            # Stats
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x,
                    self.cg_stats(env_state),
                )

                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()
            elif args.env_id in [
                "iterated_matrix_game",
            ]:
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj_1.observations,
                        traj_1.actions,
                        obs1,
                    ),
                )
            elif args.env_id == "InTheMatrix":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipditm_stats(
                        env_state,
                        traj_1,
                        traj_2,
                        args.num_envs,
                    ),
                )
            elif args.env_id == "Cournot":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    cournot_stats(
                        traj_1[0].observations, _env_params, args.num_players
                    ),
                )
            elif args.env_id == "Fishery":
                env_stats = fishery_stats(traj_1 + traj_2, args.num_players)
            elif args.env_id == "Rice-N":
                env_stats = rice_stats(
                    traj_1 + traj_2, args.num_players, args.has_mediator
                )
            elif args.env_id == "C-Rice-N":
                env_stats = c_rice_stats(
                    traj_1 + traj_2, args.num_players, args.has_mediator
                )
            else:
                env_stats = {}

            env_stats = env_stats | {
                "train/agent1_roles": agent1_roles,
                "train/agent2_roles": agent2_roles,
            }

            return (
                fitness,
                other_fitness,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
                a2_state,
            )

        self.rollout = jax.pmap(
            _rollout,
            in_axes=(0, None, None, None, None, None, None),
            static_broadcasted_argnums=6,
        )

        print(
            f"Time to Compile Jax Methods: {time.time() - self.start_time} Seconds"
        )

    def run_loop(
        self,
        env_params,
        agents,
        num_iters: int,
        watchers: Callable,
    ):
        """Run training of agents in environment"""
        print("Training")
        print("------------------------------")
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Number of Generations: {num_iters}")
        print(f"Number of Meta Episodes: {self.num_outer_steps}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {self.args.num_envs}")
        print(f"Number of Opponent: {self.args.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents[0], agents[1]
        rng, _ = jax.random.split(self.random_key)

        # Initialize evolution
        num_gens = num_iters
        strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper
        popsize = self.popsize
        num_opps = self.num_opps
        evo_state = strategy.initialize(rng, es_params)
        fit_shaper = FitnessShaper(
            maximize=self.args.es.maximise,
            centered_rank=self.args.es.centered_rank,
            w_decay=self.args.es.w_decay,
            z_score=self.args.es.z_score,
        )
        es_logging = ESLog(
            param_reshaper.total_params,
            num_gens,
            top_k=self.top_k,
            maximize=True,
        )
        log = es_logging.initialize()

        # Reshape a single agent's params before vmapping
        init_hidden = jnp.tile(
            agent1._mem.hidden,
            (popsize, num_opps, 1, 1),
        )
        a1_rng = jax.random.split(rng, popsize)
        agent1._state, agent1._mem = agent1.batch_init(
            a1_rng,
            init_hidden,
        )

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state = None

        for gen in range(num_gens):
            rng, rng_run, rng_evo, rng_key = jax.random.split(rng, 4)

            # Ask
            x, evo_state = strategy.ask(rng_evo, evo_state, es_params)
            params = param_reshaper.reshape(x)
            if self.args.num_devices == 1:
                params = jax.tree_util.tree_map(
                    lambda x: jax.lax.expand_dims(x, (0,)), params
                )

            if gen % self.args.agent2_reset_interval == 0:
                a2_state = None

            if self.args.num_devices == 1 and a2_state is not None:
                # The first rollout returns a2_state with an extra batch dim that
                # will cause issues when passing it back to the vmapped batch_policy
                a2_state = jax.tree_util.tree_map(
                    lambda w: jnp.squeeze(w, axis=0), a2_state
                )

            self_play_prob = gen / num_gens
            agent1_roles = self.args.agent1_roles
            if self.args.self_play_anneal:
                agent1_roles = np.random.binomial(
                    self.args.num_players, self_play_prob
                )
                agent1_roles = np.maximum(
                    agent1_roles, 1
                )  # Ensure at least one agent 1
            agent2_roles = self.args.num_players - agent1_roles

            # Evo Rollout
            (
                fitness,
                other_fitness,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
                a2_state,
            ) = self.rollout(
                params,
                rng_run,
                a1_state,
                a1_mem,
                a2_state,
                env_params,
                (agent1_roles, agent2_roles),
            )

            # Aggregate over devices
            fitness = jnp.reshape(
                fitness, popsize * self.args.num_devices
            ).astype(dtype=jnp.float32)
            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), env_stats)

            # Tell
            fitness_re = fit_shaper.apply(x, fitness)

            if self.args.es.mean_reduce:
                fitness_re = fitness_re - fitness_re.mean()
            evo_state = strategy.tell(x, fitness_re, evo_state, es_params)

            # Logging
            log = es_logging.update(log, x, fitness)

            is_last_loop = gen == num_iters - 1
            # Saving
            if gen % self.args.save_interval == 0 or is_last_loop:
                log_savepath1 = os.path.join(
                    self.save_dir, f"generation_{gen}"
                )
                if self.args.num_devices > 1:
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
                save(top_params, log_savepath1)
                log_savepath2 = os.path.join(
                    self.save_dir, f"agent2_iteration_{gen}"
                )
                save(a2_state.params, log_savepath2)
                if watchers:
                    print(f"Saving iteration {gen} locally and to WandB")
                    wandb.save(log_savepath1)
                    wandb.save(log_savepath2)
                else:
                    print(f"Saving iteration {gen} locally")
            if gen % log_interval == 0 or is_last_loop:
                print(f"Generation: {gen}/{num_iters}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"Fitness: {fitness.mean()} | Other Fitness: {other_fitness.mean()}"
                )
                print(
                    f"Reward Per Timestep: {float(rewards_1.mean()), float(rewards_2.mean())}"
                )
                print(
                    f"Env Stats: {jax.tree_map(lambda x: x.item(), env_stats)}"
                )
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
                    "train_iteration": gen,
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
                    "train/reward_per_timestep/player_1": float(
                        rewards_1.mean()
                    ),
                    "train/reward_per_timestep/player_2": float(
                        rewards_2.mean()
                    ),
                }
                wandb_log.update(env_stats)
                # loop through population
                for idx, (overall_fitness, gen_fitness) in enumerate(
                    zip(log["top_fitness"], log["top_gen_fitness"])
                ):
                    wandb_log[
                        f"train/fitness/top_overall_agent_{idx + 1}"
                    ] = overall_fitness
                    wandb_log[
                        f"train/fitness/top_gen_agent_{idx + 1}"
                    ] = gen_fitness

                # player 2 metrics
                # metrics [outer_timesteps, num_opps]
                flattened_metrics = {}
                if a2_metrics is not None:
                    flattened_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )

                agent2._logger.metrics.update(flattened_metrics)
                for watcher, agent in zip(watchers, agents):
                    watcher(agent)
                wandb_log = jax.tree_util.tree_map(
                    lambda x: x.item() if isinstance(x, jax.Array) else x,
                    wandb_log,
                )
                wandb.log(wandb_log)

        return agents
