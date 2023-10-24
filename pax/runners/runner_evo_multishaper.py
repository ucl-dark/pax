import os
import time
from datetime import datetime
from functools import partial
from typing import Any, Callable, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from evosax import FitnessShaper
from omegaconf import OmegaConf

import wandb
from pax.utils import MemoryState, TrainingState, save

# TODO: import when evosax library is updated
# from evosax.utils import ESLog
from pax.watchers import ESLog, n_player_ipd_visitation

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


class MultishaperEvoRunner:
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
        # TODO JIT this
        self.ipd_stats = n_player_ipd_visitation

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
        self.num_players = args.num_players
        self.num_shapers = args.num_shapers
        self.num_targets = args.num_players - args.num_shapers
        self.num_outer_steps = args.num_outer_steps
        shapers = agents[: self.num_shapers]
        targets = agents[self.num_shapers :]

        # vmap agents accordingly
        # shapers are batched over popsize and num_opps
        for shaper_agent in shapers:
            shaper_agent.batch_init = jax.vmap(
                jax.vmap(
                    shaper_agent.make_initial_state,
                    (None, 0),  # (params, rng)
                    (None, 0),  # (TrainingState, MemoryState)
                ),
                # both for Population
            )
            shaper_agent.batch_reset = jax.jit(
                jax.vmap(
                    jax.vmap(shaper_agent.reset_memory, (0, None), 0),
                    (0, None),
                    0,
                ),
                static_argnums=1,
            )

            shaper_agent.batch_policy = jax.jit(
                jax.vmap(
                    jax.vmap(shaper_agent._policy, (None, 0, 0), (0, None, 0)),
                )
            )
        # go through opponents, we start with agent2
        for agent_idx, target_agent in enumerate(targets):
            agent_arg = f"agent{agent_idx + self.num_shapers + 1}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) == "NaiveEx":
                # special case where NaiveEx has a different call signature
                target_agent.batch_init = jax.jit(
                    jax.vmap(jax.vmap(target_agent.make_initial_state))
                )
            else:
                target_agent.batch_init = jax.jit(
                    jax.vmap(
                        jax.vmap(
                            target_agent.make_initial_state, (0, None), 0
                        ),
                        (0, None),
                        0,
                    )
                )

            target_agent.batch_policy = jax.jit(
                jax.vmap(jax.vmap(target_agent._policy, 0, 0))
            )
            target_agent.batch_reset = jax.jit(
                jax.vmap(
                    jax.vmap(target_agent.reset_memory, (0, None), 0),
                    (0, None),
                    0,
                ),
                static_argnums=1,
            )

            target_agent.batch_update = jax.jit(
                jax.vmap(
                    jax.vmap(target_agent.update, (1, 0, 0, 0)),
                    (1, 0, 0, 0),
                )
            )
            if OmegaConf.select(args, agent_arg) != "NaiveEx":
                # NaiveEx requires env first step to init.
                init_hidden = jnp.tile(
                    target_agent._mem.hidden, (args.num_opps, 1, 1)
                )

                agent_rng = jnp.concatenate(
                    [
                        jax.random.split(
                            target_agent._state.random_key, args.num_opps
                        )
                    ]
                    * args.popsize
                ).reshape(args.popsize, args.num_opps, -1)

                (
                    target_agent._state,
                    target_agent._mem,
                ) = target_agent.batch_init(
                    agent_rng,
                    init_hidden,
                )

        # jit evo
        strategy.ask = jax.jit(strategy.ask)
        strategy.tell = jax.jit(strategy.tell)
        param_reshaper.reshape = jax.jit(param_reshaper.reshape)

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_reward,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ) = carry
            new_targets_mem = [None] * self.num_targets
            new_shapers_mem = [None] * self.num_shapers
            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, :, 0, :]

            # a1_rng = rngs[:, :, :, 1, :]
            # a2_rng = rngs[:, :, :, 2, :]
            rngs = rngs[:, :, :, 3, :]
            shapers_actions = []
            targets_actions = []
            for agent_idx, shaper_agent in enumerate(shapers):
                (
                    shaper_action,
                    shapers_state[agent_idx],
                    new_shapers_mem[agent_idx],
                ) = shaper_agent.batch_policy(
                    shapers_state[agent_idx],
                    shapers_obs[agent_idx],
                    shapers_mem[agent_idx],
                )
                shapers_actions.append(shaper_action)

            for agent_idx, target_agent in enumerate(targets):
                (
                    target_action,
                    targets_state[agent_idx],
                    new_targets_mem[agent_idx],
                ) = target_agent.batch_policy(
                    targets_state[agent_idx],
                    targets_obs[agent_idx],
                    targets_mem[agent_idx],
                )
                targets_actions.append(target_action)
            (
                all_agent_next_obs,
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                tuple(shapers_actions + targets_actions),
                env_params,
            )
            shapers_next_obs = all_agent_next_obs[: self.num_shapers]
            targets_next_obs = all_agent_next_obs[self.num_shapers :]
            shapers_reward, targets_reward = (
                all_agent_rewards[: self.num_shapers],
                all_agent_rewards[self.num_shapers :],
            )

            shapers_traj = [
                Sample(
                    shapers_obs[agent_idx],
                    shapers_actions[agent_idx],
                    shapers_reward[agent_idx],
                    new_shapers_mem[agent_idx].extras["log_probs"],
                    new_shapers_mem[agent_idx].extras["values"],
                    done,
                    shapers_mem[agent_idx].hidden,
                )
                for agent_idx in range(self.num_shapers)
            ]
            targets_traj = [
                Sample(
                    targets_obs[agent_idx],
                    targets_actions[agent_idx],
                    targets_rewards[agent_idx],
                    new_targets_mem[agent_idx].extras["log_probs"],
                    new_targets_mem[agent_idx].extras["values"],
                    done,
                    targets_mem[agent_idx].hidden,
                )
                for agent_idx in range(self.num_targets)
            ]
            # jax.debug.breakpoint()
            # print(len(shapers_next_obs))
            # print(len(targets_next_obs))
            # print(shapers_next_obs[0].shape)
            # print(targets_next_obs[0].shape)
            return (
                rngs,
                tuple(shapers_next_obs),
                tuple(targets_next_obs),
                tuple(shapers_reward),
                tuple(targets_reward),
                shapers_state,
                targets_state,
                new_shapers_mem,
                new_targets_mem,
                env_state,
                env_params,
            ), tuple(shapers_traj + targets_traj)

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=args.num_inner_steps,
            )
            targets_metrics = [None] * self.num_targets
            (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            for agent_idx, shaper_agent in enumerate(shapers):
                agent_arg = f"agent{agent_idx + 1}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "MFOS":
                    shapers_mem[agent_idx] = shaper_agent.meta_policy(
                        shapers_mem[agent_idx]
                    )
            # update opponents
            targets_traj = trajectories[self.num_shapers :]
            for agent_idx, target_agent in enumerate(targets):
                (
                    targets_state[agent_idx],
                    targets_mem[agent_idx],
                    targets_metrics[agent_idx],
                ) = target_agent.batch_update(
                    targets_traj[agent_idx],
                    targets_obs[agent_idx],
                    targets_state[agent_idx],
                    targets_mem[agent_idx],
                )
            return (
                rngs,
                shapers_obs,
                targets_obs,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                env_params,
            ), (trajectories, targets_metrics)

        def _rollout(
            _params: List[jnp.ndarray],
            _rng_run: jnp.ndarray,
            _shapers_state: List[TrainingState],
            _shapers_mem: List[MemoryState],
            _env_params: Any,
        ):
            # env reset
            env_rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
                * args.num_opps
                * args.popsize
            ).reshape((args.popsize, args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(env_rngs, _env_params)
            shapers_obs = obs[: self.num_shapers]
            targets_obs = obs[self.num_shapers :]
            rewards = [
                jnp.zeros((args.popsize, args.num_opps, args.num_envs)),
            ] * args.num_players

            # Shapers
            for agent_idx, shaper_agent in enumerate(shapers):
                _shapers_state[agent_idx] = _shapers_state[agent_idx]._replace(
                    params=_params[agent_idx]
                )
                _shapers_mem[agent_idx] = shaper_agent.batch_reset(
                    _shapers_mem[agent_idx], False
                )
            # Other players
            targets_mem = [None] * self.num_targets
            targets_state = [None] * self.num_targets

            _rng_run, *target_rngs = jax.random.split(
                _rng_run, self.num_players
            )
            for agent_idx, target_agent in enumerate(targets):
                # if eg 2 shapers, agent3 is the first non-shaper
                agent_arg = f"agent{agent_idx + 1 + self.num_shapers}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "NaiveEx":
                    (
                        targets_mem[agent_idx],
                        targets_state[agent_idx],
                    ) = target_agent.batch_init(targets_obs[agent_idx])
                else:
                    # meta-experiments - init 2nd agent per trial
                    target_agent_rng = jnp.concatenate(
                        [
                            jax.random.split(
                                target_rngs[agent_idx], args.num_opps
                            )
                        ]
                        * args.popsize
                    ).reshape(args.popsize, args.num_opps, -1)
                    (
                        targets_state[agent_idx],
                        targets_mem[agent_idx],
                    ) = target_agent.batch_init(
                        target_agent_rng,
                        target_agent._mem.hidden,
                    )

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    env_rngs,
                    tuple(obs[: self.num_shapers]),
                    tuple(obs[self.num_shapers :]),
                    tuple(rewards[: self.num_shapers]),
                    tuple(rewards[self.num_shapers :]),
                    _shapers_state,
                    targets_state,
                    _shapers_mem,
                    targets_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=self.num_outer_steps,
            )
            (
                env_rngs,
                shapers_obs,
                targets_obs,
                shapers_rewards,
                targets_rewards,
                shapers_state,
                targets_state,
                shapers_mem,
                targets_mem,
                env_state,
                _env_params,
            ) = vals
            trajectories, targets_metrics = stack
            shapers_traj = trajectories[: self.num_shapers]
            targets_traj = trajectories[self.num_shapers :]

            # Fitness
            shapers_fitness = [
                shapers_traj[shaper_idx].rewards.mean(axis=(0, 1, 3, 4))
                for shaper_idx in range(self.num_shapers)
            ]
            targets_fitness = [
                targets_traj[targets_idx].rewards.mean(axis=(0, 1, 3, 4))
                for targets_idx in range(self.num_targets)
            ]
            # # Stats
            if args.env_id in [
                "iterated_nplayer_tensor_game",
            ]:
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        trajectories[0].observations, args.num_players
                    ),
                )
                shapers_rewards = [
                    shapers_traj[shaper_idx].rewards.mean()
                    for shaper_idx in range(self.num_shapers)
                ]

                targets_rewards = [
                    targets_traj[target_idx].rewards.mean()
                    for target_idx in range(self.num_targets)
                ]
            else:
                raise NotImplementedError

            return (
                shapers_fitness,
                targets_fitness,
                env_stats,
                shapers_rewards,
                targets_rewards,
                targets_metrics,
            )

        self.rollout = jax.pmap(
            _rollout,
            in_axes=(0, None, None, None, None),
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
        print(f"Number of Players: {self.num_players}")
        print(f"Number of Shapers: {self.num_shapers}")
        print(f"Number of Targets: {self.num_targets}")
        print(f"Number of Generations: {num_iters}")
        print(f"Number of Meta Episodes: {self.num_outer_steps}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {self.args.num_envs}")
        print(f"Number of Opponent: {self.args.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        # Initialize agents and RNG
        rng, *evo_keys = jax.random.split(
            self.random_key, self.num_shapers + 1
        )
        # Initialize evolution
        num_gens = num_iters
        strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper
        popsize = self.popsize
        num_opps = self.num_opps
        evo_states = [
            strategy.initialize(evo_keys[idx], es_params)
            for idx in range(self.num_shapers)
        ]
        fit_shaper = FitnessShaper(
            maximize=self.args.es.maximise,
            centered_rank=self.args.es.centered_rank,
            w_decay=self.args.es.w_decay,
            z_score=self.args.es.z_score,
        )
        es_logging = [
            ESLog(
                param_reshaper.total_params,
                num_gens,
                top_k=self.top_k,
                maximize=True,
            )
        ] * self.num_shapers
        logs = [es_log.initialize() for es_log in es_logging]

        # Reshape a single agent's params before vmapping
        shaper_agents = agents[: self.num_shapers]

        init_hiddens = [
            jnp.tile(
                shaper_agents[shaper_idx]._mem.hidden,
                (popsize, num_opps, 1, 1),
            )
            for shaper_idx in range(self.num_shapers)
        ]

        shapers_state = []
        shapers_mem = []
        for shaper_idx, shaper_agent in enumerate(shaper_agents):
            rng, shaper_rng = jax.random.split(rng, 2)
            pop_shaper_rng = jax.random.split(shaper_rng, popsize)
            shaper_agent._state, shaper_agent._mem = shaper_agent.batch_init(
                pop_shaper_rng,
                init_hiddens[shaper_idx],
            )
            shapers_state.append(shaper_agent._state)
            shapers_mem.append(shaper_agent._mem)

        for gen in range(num_gens):
            rng, rng_run, rng_evo, rng_key = jax.random.split(rng, 4)
            # Ask
            xs = []
            shapers_params = []
            old_evo_states = evo_states
            evo_states = []
            for shaper_idx, _ in enumerate(shaper_agents):
                shaper_rng, rng_evo = jax.random.split(rng_evo, 2)
                x, evo_state = strategy.ask(
                    shaper_rng, old_evo_states[shaper_idx], es_params
                )
                params = param_reshaper.reshape(x)
                if self.args.num_devices == 1:
                    params = jax.tree_util.tree_map(
                        lambda x: jax.lax.expand_dims(x, (0,)), params
                    )
                evo_states.append(evo_state)
                xs.append(x)
                shapers_params.append(params)
            # Evo Rollout
            (
                shapers_fitness,
                targets_fitness,
                env_stats,
                shapers_rewards,
                targets_rewards,
                targets_metrics,
            ) = self.rollout(
                shapers_params, rng_run, shapers_state, shapers_mem, env_params
            )

            # Aggregate over devices
            shapers_fitness = [
                jnp.reshape(fitness, popsize * self.args.num_devices)
                for fitness in shapers_fitness
            ]
            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), env_stats)

            # Tell
            fitness_re = [
                fit_shaper.apply(x, fitness)
                for x, fitness in zip(xs, shapers_fitness)
            ]

            if self.args.es.mean_reduce:
                fitness_re = [fit_re - fit_re.mean() for fit_re in fitness_re]
            evo_states = [
                strategy.tell(x, fit_re, evo_state, es_params)
                for x, fit_re, evo_state in zip(xs, fitness_re, evo_states)
            ]

            # Logging
            logs = [
                es_log.update(log, x, fitness)
                for es_log, log, x, fitness in zip(
                    es_logging, logs, xs, shapers_fitness
                )
            ]
            # Saving
            if gen % self.args.save_interval == 0 or gen == num_gens - 1:
                for shaper_idx in range(self.num_shapers):
                    log_savepath = os.path.join(
                        self.save_dir, f"generation_{gen}_agent_{shaper_idx}"
                    )
                    if self.args.num_devices > 1:
                        top_params = param_reshaper.reshape(
                            logs[shaper_idx]["top_gen_params"][
                                0 : self.args.num_devices
                            ]
                        )
                        top_params = jax.tree_util.tree_map(
                            lambda x: x[0].reshape(x[0].shape[1:]), top_params
                        )
                    else:
                        top_params = param_reshaper.reshape(
                            logs[shaper_idx]["top_gen_params"][0:1]
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
                    [
                        f"Shaper{idx} fitness: {shapers_fitness[idx].mean()} "
                        for idx in range(self.num_shapers)
                    ]
                )
                print(
                    [
                        f"Target{idx} fitness: {targets_fitness[idx].mean()} "
                        for idx in range(self.num_targets)
                    ]
                )
                print(
                    f"Shapers Reward Per Timestep: {[float(reward.mean()) for reward in shapers_rewards]}"
                )
                print(
                    f"Targets Reward Per Timestep: {[float(reward.mean()) for reward in targets_rewards]}"
                )
                print(
                    f"Env Stats: {jax.tree_map(lambda x: x.item(), env_stats)}"
                )

            if watchers:
                shaper_rewards_strs = [
                    "train/reward_per_timestep/shaper_" + str(i)
                    for i in range(1, self.num_shapers + 1)
                ]
                shaper_rewards_val = [
                    float(reward.mean()) for reward in shapers_rewards
                ]
                target_rewards_strs = [
                    "train/reward_per_timestep/target_" + str(i)
                    for i in range(1, self.num_targets + 1)
                ]
                target_rewards_val = [
                    float(reward.mean()) for reward in targets_rewards
                ]
                rewards_strs = shaper_rewards_strs + target_rewards_strs
                rewards_val = shaper_rewards_val + target_rewards_val
                rewards_dict = dict(zip(rewards_strs, rewards_val))

                shaper_fitness_str = [
                    "train/fitness/shaper_" + str(i)
                    for i in range(1, self.num_shapers + 1)
                ]
                shaper_fitness_val = [
                    float(fitness.mean()) for fitness in shapers_fitness
                ]
                target_fitness_str = [
                    "train/fitness/target_" + str(i)
                    for i in range(1, self.num_targets + 1)
                ]
                target_fitness_val = [
                    float(fitness.mean()) for fitness in targets_fitness
                ]
                fitness_strs = shaper_fitness_str + target_fitness_str
                fitness_vals = shaper_fitness_val + target_fitness_val

                fitness_dict = dict(zip(fitness_strs, fitness_vals))

                shaper_welfare = float(
                    sum([reward.mean() for reward in shapers_rewards])
                    / self.num_shapers
                )
                if self.num_targets > 0:
                    target_welfare = float(
                        sum([reward.mean() for reward in targets_rewards])
                        / self.num_targets
                    )
                else:
                    target_welfare = 0

                all_rewards = shapers_rewards + targets_rewards
                global_welfare = float(
                    sum([reward.mean() for reward in all_rewards])
                    / self.args.num_players
                )
                wandb_log = {
                    "train_iteration": gen,
                    # "train/fitness/top_overall_mean": log["log_top_mean"][gen],
                    # "train/fitness/top_overall_std": log["log_top_std"][gen],
                    # "train/fitness/top_gen_mean": log["log_top_gen_mean"][gen],
                    # "train/fitness/top_gen_std": log["log_top_gen_std"][gen],
                    # "train/fitness/gen_std": log["log_gen_std"][gen],
                    "train/time/minutes": float(
                        (time.time() - self.start_time) / 60
                    ),
                    "train/time/seconds": float(
                        (time.time() - self.start_time)
                    ),
                    "train/welfare/shaper": shaper_welfare,
                    "train/welfare/target": target_welfare,
                    "train/global_welfare": global_welfare,
                } | rewards_dict
                wandb_log = wandb_log | fitness_dict
                wandb_log.update(env_stats)
                # # loop through population
                # for idx, (overall_fitness, gen_fitness) in enumerate(
                #     zip(log["top_fitness"], log["top_gen_fitness"])
                # ):
                #     wandb_log[
                #         f"train/fitness/top_overall_agent_{idx+1}"
                #     ] = overall_fitness
                #     wandb_log[
                #         f"train/fitness/top_gen_agent_{idx+1}"
                #     ] = gen_fitness

                # other player metrics
                # metrics [outer_timesteps, num_opps]
                for agent, metrics in zip(agents[1:], targets_metrics):
                    flattened_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), metrics
                    )

                    agent._logger.metrics.update(flattened_metrics)
                # TODO fix agent logger
                # for watcher, agent in zip(watchers, agents):
                # watcher(agent)
                wandb_log = jax.tree_util.tree_map(
                    lambda x: x.item() if isinstance(x, jax.Array) else x,
                    wandb_log,
                )
                wandb.log(wandb_log)

        return agents
