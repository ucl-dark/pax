import os
import time
from datetime import datetime
from typing import Any, Callable, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
from evosax import FitnessShaper
from omegaconf import OmegaConf
import wandb
from pax.utils import MemoryState, TrainingState, save

# TODO: import when evosax library is updated
# from evosax.utils import ESLog
from pax.watchers import (
    ESLog,
    third_party_punishment_visitation,
    third_party_random_visitation,
)

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


class ThirdPartyEvoRunner:
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
        self.third_party_punishment_stats = jax.jit(
            third_party_random_visitation
        )

        # Evo Runner has 3 vmap dims (popsize, num_opps, num_envs)
        # Evo Runner also has an additional pmap dim (num_devices, ...)
        # For the env we vmap over the rng but not params

        # num envs
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, 0, 0, None), 0  # rng, state, actions, params
        )

        # num opps
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, 0, 0, None), 0  # rng, state, actions, params
        )
        # pop size
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step,
                (0, 0, 0, 0, 0, None),
                0,  # rng, state, actions, params
            )
        )
        self.split = jax.vmap(
            jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None)),
            (0, None),
        )

        self.num_outer_steps = args.num_outer_steps
        agent1, *other_agents = agents

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
        # go through opponents, we start with agent2
        for agent_idx, non_first_agent in enumerate(other_agents):
            agent_arg = f"agent{agent_idx+2}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) == "NaiveEx":
                # special case where NaiveEx has a different call signature
                non_first_agent.batch_init = jax.jit(
                    jax.vmap(jax.vmap(non_first_agent.make_initial_state))
                )
            else:
                non_first_agent.batch_init = jax.jit(
                    jax.vmap(
                        jax.vmap(
                            non_first_agent.make_initial_state, (0, None), 0
                        ),
                        (0, None),
                        0,
                    )
                )

            non_first_agent.batch_policy = jax.jit(
                jax.vmap(jax.vmap(non_first_agent._policy, 0, 0))
            )
            non_first_agent.batch_reset = jax.jit(
                jax.vmap(
                    jax.vmap(non_first_agent.reset_memory, (0, None), 0),
                    (0, None),
                    0,
                ),
                static_argnums=1,
            )

            non_first_agent.batch_update = jax.jit(
                jax.vmap(
                    jax.vmap(non_first_agent.update, (1, 0, 0, 0)),
                    (1, 0, 0, 0),
                )
            )
            if OmegaConf.select(args, agent_arg) != "NaiveEx":
                # NaiveEx requires env first step to init.
                init_hidden = jnp.tile(
                    non_first_agent._mem.hidden, (args.num_opps, 1, 1)
                )

                agent_rng = jnp.concatenate(
                    [
                        jax.random.split(
                            non_first_agent._state.random_key, args.num_opps
                        )
                    ]
                    * args.popsize
                ).reshape(args.popsize, args.num_opps, -1)

                (
                    non_first_agent._state,
                    non_first_agent._mem,
                ) = non_first_agent.batch_init(
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
                prev_actions,
                prev_player_selection,
                obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = carry
            new_other_agent_mem = [None] * len(other_agents)
            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, :, 0, :]
            # a1_rng = rngs[:, :, :, 1, :]
            # a2_rng = rngs[:, :, :, 2, :]

            rngs = rngs[:, :, :, 3, :]

            actions = []
            (
                first_action,
                first_agent_state,
                new_first_agent_mem,
            ) = agent1.batch_policy(
                first_agent_state,
                obs,
                first_agent_mem,
            )
            actions.append(first_action)
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    non_first_action,
                    other_agent_state[agent_idx],
                    new_other_agent_mem[agent_idx],
                ) = non_first_agent.batch_policy(
                    other_agent_state[agent_idx],
                    obs,
                    other_agent_mem[agent_idx],
                )
                actions.append(non_first_action)
            (
                player_selection,
                (next_obs, log_obs),
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                prev_actions,
                prev_player_selection,
                actions,
                env_params,
            )

            first_agent_reward, *other_agent_rewards = all_agent_rewards

            traj1 = Sample(
                obs,
                first_action,
                first_agent_reward,
                new_first_agent_mem.extras["log_probs"],
                new_first_agent_mem.extras["values"],
                done,
                first_agent_mem.hidden,
            )
            other_traj = [
                Sample(
                    obs,
                    actions[agent_idx + 1],
                    other_agent_rewards[agent_idx],
                    new_other_agent_mem[agent_idx].extras["log_probs"],
                    new_other_agent_mem[agent_idx].extras["values"],
                    done,
                    other_agent_mem[agent_idx].hidden,
                )
                for agent_idx in range(len(other_agents))
            ]
            return (
                rngs,
                actions,  # this are the next prev actions
                player_selection,  # this is the player selection for above actions
                next_obs,
                first_agent_reward,
                tuple(other_agent_rewards),
                first_agent_state,
                other_agent_state,
                new_first_agent_mem,
                new_other_agent_mem,
                env_state,
                env_params,
            ), (log_obs, traj1, *other_traj)

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, stack = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=args.num_inner_steps,
            )
            log_obs = stack[0]
            trajectories = stack[1:]
            other_agent_metrics = [None] * len(other_agents)
            (
                rngs,
                _1,
                _2,
                obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                first_agent_mem = agent1.meta_policy(first_agent_mem)
            # update opponents, we start with agent2
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                    other_agent_metrics[agent_idx],
                ) = non_first_agent.batch_update(
                    trajectories[agent_idx + 1],
                    obs,
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                )
            return (
                rngs,
                _1,
                _2,
                obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ), (log_obs, trajectories, other_agent_metrics)

        def _rollout(
            _params: jnp.ndarray,
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _env_params: Any,
        ):
            # env reset
            (
                _rng_run,
                first_action_rng1,
                first_action_rng2,
                first_action_rng3,
            ) = jax.random.split(_rng_run, 4)
            env_rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
                * args.num_opps
                * args.popsize
            ).reshape((args.popsize, args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(env_rngs, _env_params)
            rewards = [
                jnp.zeros((args.popsize, args.num_opps, args.num_envs)),
            ] * args.num_players
            # first action is uniform random C or D with no punishment - so action 0 or 4
            first_action1 = (
                jax.random.randint(
                    first_action_rng1,
                    (args.popsize, args.num_opps, args.num_envs),
                    0,
                    2,
                )
                * 4
            )
            first_action2 = (
                jax.random.randint(
                    first_action_rng2,
                    (args.popsize, args.num_opps, args.num_envs),
                    0,
                    2,
                )
                * 4
            )
            first_action3 = (
                jax.random.randint(
                    first_action_rng3,
                    (args.popsize, args.num_opps, args.num_envs),
                    0,
                    2,
                )
                * 4
            )
            first_action = [first_action1, first_action2, first_action3]
            _rng_run, player_sel_rng = jax.random.split(
                _rng_run,
            )

            # first_player_selections is indep permutation of arange(0,3) with shape (popsize, num_opps, num_envs,3
            player_ids = jnp.tile(
                jnp.arange(3), (args.popsize, args.num_opps, args.num_envs, 1)
            )
            first_player_selections = jax.random.permutation(
                key=player_sel_rng, x=player_ids, axis=-1, independent=True
            )

            # Player 1
            _a1_state = _a1_state._replace(params=_params)
            _a1_mem = agent1.batch_reset(_a1_mem, False)
            # Other players
            other_agent_mem = [None] * len(other_agents)
            other_agent_state = [None] * len(other_agents)

            _rng_run, *other_agent_rngs = jax.random.split(
                _rng_run, args.num_players
            )

            for agent_idx, non_first_agent in enumerate(other_agents):
                # indexing starts at 2 for args
                agent_arg = f"agent{agent_idx+2}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "NaiveEx":
                    (
                        other_agent_mem[agent_idx],
                        other_agent_state[agent_idx],
                    ) = non_first_agent.batch_init(obs)
                else:
                    # meta-experiments - init 2nd agent per trial
                    non_first_agent_rng = jnp.concatenate(
                        [
                            jax.random.split(
                                other_agent_rngs[agent_idx], args.num_opps
                            )
                        ]
                        * args.popsize
                    ).reshape(args.popsize, args.num_opps, -1)
                    (
                        other_agent_state[agent_idx],
                        other_agent_mem[agent_idx],
                    ) = non_first_agent.batch_init(
                        non_first_agent_rng,
                        non_first_agent._mem.hidden,
                    )

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    env_rngs,
                    first_action,
                    first_player_selections,
                    obs,
                    rewards[0],
                    tuple(rewards[1:]),
                    _a1_state,
                    other_agent_state,
                    _a1_mem,
                    other_agent_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=self.num_outer_steps,
            )
            (
                env_rngs,
                _1,
                _2,
                obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                _env_params,
            ) = vals
            log_obs, trajectories, other_agent_metrics = stack

            # Fitness
            fitness = trajectories[0].rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = [
                traj.rewards.mean(axis=(0, 1, 3, 4))
                for traj in trajectories[1:]
            ]
            # # Stats
            if args.env_id in [
                "third_party_random",
            ]:
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.third_party_punishment_stats(
                        log_obs,
                    ),
                )
                first_agent_reward = trajectories[0].rewards.mean()
                other_agent_rewards = [
                    traj.rewards.mean() for traj in trajectories[1:]
                ]
            else:
                raise NotImplementedError

            return (
                fitness,
                other_fitness,
                env_stats,
                first_agent_reward,
                other_agent_rewards,
                other_agent_metrics,
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
        print(f"Number of Generations: {num_iters}")
        print(f"Number of Meta Episodes: {self.num_outer_steps}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {self.args.num_envs}")
        print(f"Number of Opponent: {self.args.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        # Initialize agents and RNG
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
            agents[0]._mem.hidden,
            (popsize, num_opps, 1, 1),
        )
        a1_rng = jax.random.split(rng, popsize)
        agents[0]._state, agents[0]._mem = agents[0].batch_init(
            a1_rng,
            init_hidden,
        )

        a1_state, a1_mem = agents[0]._state, agents[0]._mem

        for gen in range(num_gens):
            rng, rng_run, rng_evo, rng_key = jax.random.split(rng, 4)

            # Ask
            x, evo_state = strategy.ask(rng_evo, evo_state, es_params)
            params = param_reshaper.reshape(x)
            if self.args.num_devices == 1:
                params = jax.tree_util.tree_map(
                    lambda x: jax.lax.expand_dims(x, (0,)), params
                )
            # Evo Rollout
            (
                fitness,
                other_fitness,
                env_stats,
                first_agent_reward,
                other_agent_reward,
                other_agent_metrics,
            ) = self.rollout(params, rng_run, a1_state, a1_mem, env_params)

            # Aggregate over devices
            fitness = jnp.reshape(fitness, popsize * self.args.num_devices)
            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), env_stats)

            # Tell
            fitness_re = fit_shaper.apply(x, fitness)

            if self.args.es.mean_reduce:
                fitness_re = fitness_re - fitness_re.mean()
            evo_state = strategy.tell(x, fitness_re, evo_state, es_params)

            # Logging
            log = es_logging.update(log, x, fitness)

            # Saving
            if gen % self.args.save_interval == 0 or gen == num_gens - 1:
                log_savepath = os.path.join(self.save_dir, f"generation_{gen}")
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
                    f"Fitness: {fitness.mean()} | Other Fitness: {[fitness.mean() for fitness in other_fitness]}"
                )
                print(
                    f"Reward Per Timestep: {float(first_agent_reward.mean()), *[float(reward.mean()) for reward in other_agent_reward]}"
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
                rewards_strs = [
                    "train/reward_per_timestep/player_" + str(i)
                    for i in range(2, len(other_agent_reward) + 2)
                ]
                rewards_val = [
                    float(reward.mean()) for reward in other_agent_reward
                ]
                rewards_dict = dict(zip(rewards_strs, rewards_val))
                fitness_str = [
                    "train/fitness/player_" + str(i)
                    for i in range(2, len(other_fitness) + 2)
                ]
                fitness_val = [
                    float(fitness.mean()) for fitness in other_fitness
                ]
                fitness_dict = dict(zip(fitness_str, fitness_val))
                all_rewards = other_agent_reward + [first_agent_reward]
                global_welfare = float(
                    sum([reward.mean() for reward in all_rewards])
                    / self.args.num_players
                )
                wandb_log = {
                    "train_iteration": gen,
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
                    "train/fitness/player_1": float(fitness.mean()),
                    "train/reward_per_timestep/player_1": float(
                        first_agent_reward.mean()
                    ),
                    "train/global_welfare": global_welfare,
                } | rewards_dict
                wandb_log = wandb_log | fitness_dict
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

                # other player metrics
                # metrics [outer_timesteps, num_opps]
                for agent, metrics in zip(agents[1:], other_agent_metrics):
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
