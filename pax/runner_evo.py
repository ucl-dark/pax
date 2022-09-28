from datetime import datetime
import os
import time
from typing import NamedTuple

from evosax import FitnessShaper
import jax
import jax.numpy as jnp
import wandb
from pax.utils import save, TrainingState, MemoryState

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
        self.cg_stats = cg_visitation

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

            # MFOS has to takes a meta-action for each episode
            if self.args.agent1 == "MFOS":
                a1_mem = a1_mem._replace(th=a1_mem.curr_th)

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

        def evo_rollout(
            params: jnp.ndarray,
            rng_run: jnp.ndarray,
            rng_key: jnp.ndarray,
            a1_state: TrainingState,
            a1_mem: MemoryState,
        ):
            # env reset
            t_init, env_state = env.runner_reset(
                (popsize, num_opps, env.num_envs), rng_run
            )
            # Player 1
            a1_state = a1_state._replace(params=params)
            a1_mem = agent1.batch_reset(a1_mem, False)
            # Player 2
            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(t_init[1])

            else:
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng_key, popsize * num_opps).reshape(
                        self.popsize, num_opps, -1
                    ),
                    agent2._mem.hidden,
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

            # Stats
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
            return (
                fitness,
                other_fitness,
                env_stats,
                rewards_0,
                rewards_1,
                a2_metrics,
            )

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

        # Evolution specific: add pop size dimension
        if self.args.env_type == "infinite" and self.args.env_id == "ipd":
            env.batch_step = jax.jit(
                jax.vmap(env.batch_step, (0, None), (0, None))
            )
        else:
            env.batch_step = jax.jit(
                jax.vmap(env.batch_step),
            )

        if self.args.env_type == "coin_game":
            env.batch_reset = jax.jit(jax.vmap(env.batch_reset))

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
        evo_rollout = jax.pmap(
            evo_rollout,
            in_axes=(0, None, None, None, None),
        )
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
                rewards_0,
                rewards_1,
                a2_metrics,
            ) = evo_rollout(params, rng_run, rng_key, a1_state, a1_mem)

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
                top_params = param_reshaper.reshape_single_net(
                    evo_state.best_member
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
                    "train/episode_reward/player_1": float(rewards_0.mean()),
                    "train/episode_reward/player_2": float(rewards_1.mean()),
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
                agent1._logger.metrics.update(flattened_metrics)
                agents.log(watchers)
                wandb.log(wandb_log)

        return agents
