import os

import time
from datetime import datetime
from typing import NamedTuple

import jax
import jax.numpy as jnp
from evosax import FitnessShaper

import wandb
from pax.utils import save
from pax.ppo.ppo import make_agent

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


class SampleMFOS(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray
    meta_actions: jnp.ndarray


@jax.jit
def reduce_mfos_traj(traj: SampleMFOS) -> SampleMFOS:
    """Used to collapse lax.scan outputs dims"""
    # x: [num_devices, outer_loop, inner_loop, num_opps, num_envs ...]
    # x: [num_devices, timestep, batch_size, ...]
    num_devices = traj.observations.shape[0]
    num_envs = traj.observations.shape[3] * traj.observations.shape[4]
    num_timesteps = traj.observations.shape[1] * traj.observations.shape[2]
    reduced_traj = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (num_devices, num_timesteps, num_envs) + x.shape[5:]
        ),
        traj,
    )
    # [timesteps, num_devices, batch_size ..]
    traj = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1),
        reduced_traj,
    )
    return jax.tree_util.tree_map(
        lambda x: x.reshape(
            (num_timesteps, num_envs * num_devices) + x.shape[2:]
        ),
        reduced_traj,
    )


@jax.jit
def reduce_outer_traj(traj: Sample) -> Sample:
    """Used to collapse lax.scan outputs dims"""
    # x: [outer_loop, inner_loop, num_opps, num_envs ...]
    # x: [timestep, batch_size, ...]
    num_envs = traj.observations.shape[2] * traj.observations.shape[3]
    num_timesteps = traj.observations.shape[0] * traj.observations.shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_timesteps, num_envs) + x.shape[4:]),
        traj,
    )


class MetaSPEvoRunner:
    """Holds the runner's state."""

    def __init__(
        self,
        args,
        strategy,
        es_params,
        param_reshaper1,
        param_reshaper2,
        save_dir,
    ):
        self.algo = args.es.algo
        self.args = args
        self.es_params = es_params
        self.generations = 0
        self.num_opps = args.num_opps
        self.eval_steps = 0
        self.eval_episodes = 0
        self.param_reshaper1 = param_reshaper1
        self.param_reshaper2 = param_reshaper2
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

        def _reshape_opp_dim(x):
            # x: [num_opps, num_envs ...]
            # x: [batch_size, ...]
            batch_size = args.num_envs * args.num_opps
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), x
            )

        def pmap_reshape(cg_state):
            return jax.tree_util.tree_map(
                lambda x: x.reshape(
                    (self.num_devices * self.popsize,) + x.shape[2:]
                ),
                cg_state,
            )

        self.pmap_reshape = pmap_reshape
        self._reshape_opp_dim = _reshape_opp_dim

    # flake8: noqa: C901
    def train_loop(self, env, agents, num_generations, watchers):
        """Run training of agents in environment"""

        def _inner_nl_rollout(carry, unused):
            """Runner for inner episode"""
            t1, t3, a1_state, a1_mem, a3_state, a3_mem, env_state = carry

            # as this is self-play agent1 methods and agent2 methods are the same (and functional!)
            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                t1.observation,
                a1_mem,
            )

            a3, a3_state, new_a3_mem = agent3.batch_policy(
                a3_state,
                t3.observation,
                a3_mem,
            )

            (tprime_1, tprime_3), env_state = env.batch_step(
                (a1, a3),
                env_state,
            )

            if self.args.agent1 == "MFOS":
                traj1 = SampleMFOS(
                    t1.observation,
                    a1,
                    tprime_1.reward,
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    tprime_1.last(),
                    a1_mem.hidden,
                    a1_mem.th,
                )
            else:
                traj1 = Sample(
                    t1.observation,
                    a1,
                    tprime_1.reward,
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    tprime_1.last(),
                    a1_mem.hidden,
                )
            traj3 = Sample(
                t3.observation,
                a3,
                tprime_3.reward,
                new_a3_mem.extras["log_probs"],
                new_a3_mem.extras["values"],
                tprime_3.last(),
                a3_mem.hidden,
            )
            return (
                tprime_1,
                tprime_3,
                a1_state,
                new_a1_mem,
                a3_state,
                new_a3_mem,
                env_state,
            ), (
                traj1,
                traj3,
            )

        def _inner_meta_rollout(carry, unused):
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

            if self.args.agent1 == "MFOS":
                traj1 = SampleMFOS(
                    t1.observation,
                    a1,
                    tprime_1.reward,
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    tprime_1.last(),
                    a1_mem.hidden,
                    a1_mem.th,
                )
            else:
                traj1 = Sample(
                    t1.observation,
                    a1,
                    tprime_1.reward,
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    tprime_1.last(),
                    a1_mem.hidden,
                )
            if self.args.agent2 == "MFOS":
                traj2 = SampleMFOS(
                    t2.observation,
                    a2,
                    tprime_2.reward,
                    new_a2_mem.extras["log_probs"],
                    new_a2_mem.extras["values"],
                    tprime_2.last(),
                    a2_mem.hidden,
                    a2_mem.th,
                )
            else:
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

        def _outer_nl_rollout(carry, unused):
            """Runner nl algos"""
            t1, t2, a1_state, a1_mem, a3_state, a3_mem, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_nl_rollout,
                carry,
                None,
                length=env.inner_episode_length,
            )

            # update second agent
            t1, t3, a1_state, a1_mem, a3_state, a3_mem, env_state = vals

            # do second agent update
            final_t3 = t3._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a3_state, a3_mem, _ = agent3.batch_update(
                trajectories[1], final_t3, a3_state, a3_mem
            )
            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a3_state,
                a3_mem,
                env_state,
            ), (*trajectories, a3_state)

        def _outer_meta_rollout(carry, unused):
            """Runner meta algos"""
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_meta_rollout,
                carry,
                None,
                length=env.inner_episode_length,
            )

            # update second agent
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals

            # do second agent update
            if self.args.agent1 == "MFOS":
                a1_mem = a1_mem._replace(th=a1_mem.curr_th)

            if self.args.agent2 == "MFOS":
                a2_mem = a2_mem._replace(th=a2_mem.curr_th)

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
            ), trajectories

        def _xp_rollout(params: jnp.ndarray, rng_device: jnp.ndarray):
            """Runner for a single Naive Learner Step"""

            rng, rng_run, rng_gen, rng_key = jax.random.split(rng_device, 4)
            # this will be serialized so pulls out initial state
            a1_state, a1_mem = agent1._state, agent1._mem
            a1_state = a1_state._replace(
                params=params,
            )
            a1_mem = agent1.batch_reset(a1_mem, False)

            # init agent 2
            a3_state, a3_mem = agent3.batch_init(
                jax.random.split(rng_key, popsize * num_opps).reshape(
                    self.popsize, num_opps, -1
                ),
                agent3._mem.hidden,
            )

            # run the training loop!
            t_init, env_state = env.runner_reset(
                (popsize, num_opps, env.num_envs), rng_run
            )
            vals, stack = jax.lax.scan(
                _outer_nl_rollout,
                (*t_init, a1_state, a1_mem, a3_state, a3_mem, env_state),
                None,
                length=env.num_trials,
            )

            traj_1, traj_3, a3_metrics = stack
            t1, t3, a1_state, a1_mem, a3_state, a3_mem, env_state = vals

            # Fitness
            fitness = traj_1.rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = traj_3.rewards.mean(axis=(0, 1, 3, 4))
            if self.args.env_type == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.cg_stats(env_state),
                )
                rewards_0 = traj_1.rewards.sum(axis=1).mean()
                rewards_3 = traj_3.rewards.sum(axis=1).mean()

            elif self.args.env_type in [
                "meta",
                "sequential",
            ]:
                final_t1 = t1._replace(
                    step_type=2 * jnp.ones_like(t1.step_type)
                )
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj_1.observations,
                        traj_1.actions,
                        final_t1.observation,
                    ),
                )
                rewards_0 = traj_1.rewards.mean()
                rewards_3 = traj_3.rewards.mean()
            else:
                rewards_0 = traj_1.rewards.mean()
                rewards_3 = traj_3.rewards.mean()
                env_stats = {}

            rewards_0 = traj_1.rewards.sum(axis=1).mean()
            rewards_3 = traj_3.rewards.sum(axis=1).mean()

            return (
                fitness,
                other_fitness,
                env_stats,
                rewards_0,
                rewards_3,
            )

        def _sp_rollout(
            params1: jnp.ndarray, params2: jnp.ndarray, rng_device: jnp.ndarray
        ):
            """Runner for one fitness step"""
            (
                rng,
                rng_run,
                rng_gen,
                rng_key,
            ) = jax.random.split(rng_device, 4)
            # this will be serialized so pulls out initial state
            a1_state, a1_mem = agent1._state, agent1._mem
            a1_state = a1_state._replace(
                params=params1,
            )
            a1_mem = agent1.batch_reset(a1_mem, False)

            # init agent 2
            a2_state, a2_mem = agent2._state, agent2._mem
            a2_state = a2_state._replace(
                params=params2,
            )
            a2_mem = agent2.batch_reset(a2_mem, False)

            # run the training loop!
            t_init, env_state = env.runner_reset(
                (popsize, num_opps, env.num_envs), rng_run
            )
            vals, stack = jax.lax.scan(
                _outer_meta_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )
            traj_1, traj_2 = stack
            t1, t2, _, _, _, _, env_state = vals

            # Fitness
            fitness = traj_1.rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = traj_2.rewards.mean(axis=(0, 1, 3, 4))
            if self.args.env_type == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.cg_stats(env_state),
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
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj_1.observations,
                        traj_1.actions,
                        final_t1.observation,
                    ),
                )
                rewards_0 = traj_1.rewards.mean()
                rewards_1 = traj_2.rewards.mean()
            else:
                rewards_0 = traj_1.rewards.mean()
                rewards_1 = traj_2.rewards.mean()
                env_stats = {}

            rewards_0 = traj_1.rewards.sum(axis=1).mean()
            rewards_1 = traj_2.rewards.sum(axis=1).mean()
            return (
                fitness,
                other_fitness,
                env_stats,
                rewards_0,
                rewards_1,
            )

        def _pmap_rollout(
            a1_params: jnp.ndarray, a2_params: jnp.ndarray, rng: jnp.ndarray
        ):
            rng_sp, rng_xp_1, rng_xp_2 = jax.random.split(rng, 3)
            (
                fitness,
                _,
                env_stats_xp_1,
                rewards_0_xp,
                _,
            ) = _xp_rollout(a1_params, rng_xp_1)
            other_fitness, _, env_stats_xp_2, rewards_1_xp, _ = _xp_rollout(
                a2_params, rng_xp_2
            )
            (
                sp_fitness,
                sp_other_fitness,
                env_stats_sp,
                rewards_0_sp,
                rewards_1_sp,
            ) = _sp_rollout(a1_params, a2_params, rng_sp)

            return {
                "xp_fitness": fitness,
                "xp_other_fitness": other_fitness,
                "sp_fitness": sp_fitness,
                "sp_other_fitness": sp_other_fitness,
                "reward_0_xp": rewards_0_xp,
                "reward_1_xp": rewards_1_xp,
                "reward_0_sp": rewards_0_sp,
                "reward_1_sp": rewards_1_sp,
                "env_stats_xp_1": env_stats_xp_1,
                "env_stats_xp_2": env_stats_xp_2,
                "env_stats_sp": env_stats_sp,
            }

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
        param_reshaper1 = self.param_reshaper1
        param_reshaper2 = self.param_reshaper2
        popsize = self.popsize
        num_opps = self.num_opps
        evo_state1 = strategy.initialize(rng, es_params)
        evo_state2 = strategy.initialize(rng, es_params)
        fit_shaper = FitnessShaper(maximize=True)
        es_logging = ESLog(
            param_reshaper1.total_params,
            num_gens,
            top_k=self.top_k,
            maximize=True,
        )
        log = es_logging.initialize()

        # Pmap specific: add num_devices dimension
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step),
        )

        if self.args.env_type == "coin_game":
            env.batch_reset = jax.jit(jax.vmap(env.batch_reset))

        # these are batched
        dumb_rng = agent1._state.random_key
        vmap_state1, _ = agent1.batch_init(
            jax.random.split(dumb_rng, num_devices * popsize),
            jnp.tile(
                agent1._mem.hidden,
                (num_devices * popsize, num_opps, 1, 1),
            ),
        )
        dumb_rng = agent2._state.random_key
        vmap_state2, _ = agent2.batch_init(
            jax.random.split(dumb_rng, num_devices * popsize),
            jnp.tile(
                agent2._mem.hidden,
                (num_devices * popsize, num_opps, 1, 1),
            ),
        )

        # hiddens are used only on device (so not batched)
        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(dumb_rng, popsize),
            jnp.tile(
                agent1._mem.hidden,
                (popsize, num_opps, 1, 1),
            ),
        )

        agent2._state, agent2._mem = agent2.batch_init(
            jax.random.split(dumb_rng, popsize),
            jnp.tile(
                agent2._mem.hidden,
                (popsize, num_opps, 1, 1),
            ),
        )

        agent1._state = agent1._state._replace(params=vmap_state1.params)
        agent2._state = agent2._state._replace(params=vmap_state2.params)

        # need some naive learners!

        obs_spec = (
            env.observation_spec().shape
            if self.args.env_type == "coin_game"
            else (env.observation_spec().num_values,)
        )
        agent3 = make_agent(
            self.args,
            obs_spec=obs_spec,
            action_spec=env.action_spec().num_values,
            seed=0,
            player_id=3,
            has_sgd_jit=True,
        )

        # vmap them up
        agent3.batch_init = jax.jit(
            jax.vmap(
                jax.vmap(agent3.make_initial_state, (0, None), 0),
                (0, None),
                0,
            )
        )

        agent3.batch_policy = jax.jit(jax.vmap(jax.vmap(agent3._policy, 0, 0)))

        agent3.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent3.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent3.batch_update = jax.jit(
            jax.vmap(
                jax.vmap(agent3.update, (1, 0, 0, 0)),
                (1, 0, 0, 0),
            )
        )

        # NaiveEx requires env first step to init.
        a3_hidden = jnp.tile(agent3._mem.hidden, (self.args.num_opps, 1, 1))

        a3_key = jax.random.split(
            agent3._state.random_key, self.args.popsize * self.args.num_opps
        ).reshape(self.args.popsize, self.args.num_opps, -1)

        agent3._state, agent3._mem = agent3.batch_init(
            a3_key,
            a3_hidden,
        )
        pmap_rollout = jax.pmap(_pmap_rollout)

        lmbda = 1
        lmbda_anneal = self.args.lambda_anneal

        for gen in range(num_gens):
            # run generation step
            rng, rng_evo, rng_devices, rng_flag = jax.random.split(rng, 4)
            rng_devices = jnp.tile(rng_devices, num_devices).reshape(
                num_devices, -1
            )
            # Ask for params
            rng_evo1, rng_evo2 = jax.random.split(rng_evo, 2)

            x, evo_state1 = strategy.ask(
                rng_evo1, evo_state1, es_params
            )  # this means that x isn't of shape (total_popsize, params)
            a1_params = param_reshaper1.reshape(x)

            y, evo_state2 = strategy.ask(
                rng_evo2, evo_state2, es_params
            )  # this means that x isn't of shape (total_popsize, params)
            a2_params = param_reshaper2.reshape(y)
            if self.num_devices == 1:
                # reshape x into (num_devices, popsize, ....)
                a1_params = jax.tree_util.tree_map(
                    lambda x: jax.lax.expand_dims(x, (0,)), a1_params
                )

                a2_params = jax.tree_util.tree_map(
                    lambda x: jax.lax.expand_dims(x, (0,)), a2_params
                )

            results = pmap_rollout(a1_params, a2_params, rng_devices)
            fitness = (1 - lmbda) * results["sp_fitness"] + lmbda * results[
                "xp_fitness"
            ]
            other_fitness = (1 - lmbda) * results[
                "sp_other_fitness"
            ] + lmbda * results["xp_other_fitness"]

            # Maximize fitness
            fitness_re = fit_shaper.apply(
                x, jnp.reshape(fitness, popsize * num_devices)
            )
            other_fitness_re = fit_shaper.apply(
                y, jnp.reshape(other_fitness, popsize * num_devices)
            )

            # Tell
            evo_state1 = strategy.tell(
                x, fitness_re - fitness_re.mean(), evo_state1, es_params
            )
            evo_state2 = strategy.tell(
                y,
                other_fitness_re - other_fitness_re.mean(),
                evo_state2,
                es_params,
            )

            # Logging
            log = es_logging.update(
                log, x, jnp.reshape(fitness, popsize * num_devices)
            )

            # Saving
            if self.args.save and gen % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"generation_{gen}")
                if self.args.num_devices > 1:
                    top_params = param_reshaper1.reshape(
                        log["top_gen_params"][0 : self.args.num_devices]
                    )
                    top_params = jax.tree_util.tree_map(
                        lambda x: x[0].reshape(x[0].shape[1:]), top_params
                    )
                else:
                    top_params = param_reshaper1.reshape(
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
                if self.args.env_type == "coin_game":
                    rewards_0 = results["reward_0_sp"].mean()
                    rewards_1 = results["reward_1_sp"].mean()
                    rewards_2 = results["reward_0_xp"].mean()
                    rewards_3 = results["reward_1_xp"].mean()

                elif self.args.env_type in [
                    "meta",
                    "sequential",
                ]:
                    rewards_0 = results["reward_0_sp"].mean()
                    rewards_1 = results["reward_1_sp"].mean()
                    rewards_2 = results["reward_0_xp"].mean()
                    rewards_3 = results["reward_1_xp"].mean()
                    env_stats = results["env_stats_sp"]

                else:
                    env_stats = results["env_stats_sp"]

                print(f"Generation: {gen}| Lambda: {lmbda}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"Fitness: {fitness.mean()} | Other Fitness: {other_fitness.mean()}"
                )
                print(
                    f"Total Episode Reward SP: {float(rewards_0.mean()), float(rewards_1.mean())}"
                )
                print(
                    f"Total Episode Reward XP: {float(rewards_2.mean()), float(rewards_3.mean())}"
                )
                print(f"Env Stats SP: {results['env_stats_sp']}")
                # print(f"Env Stats XP1: {results['env_stats_xp_1']}")
                # print(f"Env Stats XP2: {results['env_stats_xp_2']}")

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
                        "generations": self.generations,
                        f"train/fitness/player_1": float(fitness.mean()),
                        f"train/fitness/player_2": float(other_fitness.mean()),
                        f"train/fitness/top_overall_mean": log["log_top_mean"][
                            gen
                        ],
                        f"train/fitness/top_overall_std": log["log_top_std"][
                            gen
                        ],
                        f"train/fitness/top_gen_mean": log["log_top_gen_mean"][
                            gen
                        ],
                        f"train/fitness/top_gen_std": log["log_top_gen_std"][
                            gen
                        ],
                        f"train/fitness/gen_std": log["log_gen_std"][gen],
                        f"train/time/minutes": float(
                            (time.time() - self.start_time) / 60
                        ),
                        "train/time/seconds": float(
                            (time.time() - self.start_time)
                        ),
                        f"train/episode_reward/player_1": float(
                            rewards_0.mean()
                        ),
                        f"train/episode_reward/player_2": float(
                            rewards_1.mean()
                        ),
                        f"train/episode_reward/player_3": float(
                            rewards_2.mean()
                        ),
                        f"train/episode_reward/player_4": float(
                            rewards_3.mean()
                        ),
                        f"train/lambda": lmbda,
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
                        lambda x: jnp.sum(jnp.mean(x, 1)), {}
                    )
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flattened_metrics
                    )

                    agent1._logger.metrics = (
                        agent1._logger.metrics | flattened_metrics
                    )
                    agents.log(watchers)
                    wandb.log(wandb_log)
            self.generations += 1
            if lmbda > 0:
                lmbda -= lmbda_anneal

        return agents
