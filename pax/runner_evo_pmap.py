from datetime import datetime
import os
import time
from typing import List, NamedTuple

from dm_env import TimeStep
from evosax import FitnessShaper
import jax
import jax.numpy as jnp
import wandb

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

    def train_loop(self, env, agents, num_episodes, watchers):
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

        def rollout(
            batched_rng: jnp.ndarray, env, a1_state, a1_mem, a2_state, a2_mem
        ) -> jnp.ndarray:
            """
            PMAPs the population and sends it to multiple devices

            Input
            batched_rng: jnp.ndarray, tiled random key
            env: CoinGame: Coin Game environment
            a1_state: TrainingState, Named Tuple: holds the training state of agent 1
            a1_mem: MemoryState, Named Tuple: holds the memory state of agent 1
            a1_state: TrainingState, Named Tuple: holds the training state of agent 2
            a1_mem: MemoryState, Named Tuple: holds the memory state of agent 2

            Output
            fitness: jnp.ndarray, (popsize, ), fitness of every agent from each device
            other_fitness: ...
            traj_1: ...
            traj_2: ...
            a2_metrics: ...
            t1: ...
            t2: ...
            env_state: ...
            """

            rng_key, rng_run = jax.random.split(batched_rng)
            t_init, env_state = env.runner_reset(
                (popsize, num_opps, env.num_envs), rng_run
            )
            # Player 2
            key = jax.random.split(rng_key, popsize * num_opps).reshape(
                self.popsize, num_opps, -1
            )
            a2_state, a2_mem = agent2.batch_init(
                key,
                a2_mem.hidden,
            )

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            traj_1, traj_2, a2_metrics = stack
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals

            fitness = traj_1.rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = traj_2.rewards.mean(axis=(0, 1, 3, 4))

            return (
                fitness,
                other_fitness,
                traj_1,
                traj_2,
                a2_metrics,
                t1,
                t2,
                env_state,
            )

        print("Training")
        print("------------------------------")
        num_iters = max(
            int(num_episodes / (self.popsize * env.num_envs * self.num_opps)),
            1,
        )
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Number of Generations: {num_iters}")
        print(f"Number of Meta Episodes: {num_episodes}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {env.num_envs}")
        print(f"Number of Opponent: {self.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)
        num_devices = self.args.num_devices

        # Initialize evolution
        num_gens = num_episodes
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
        pmapped_rollout = jax.pmap(rollout)

        # Evolution specific: add pop size dimension
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
        # key = (2, 2)
        # init_hidden = (2, 1, 1000, 16)
        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(agent1._state.random_key, popsize),
            init_hidden,
        )

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        for gen in range(num_iters):
            rng, rng_gen, rng_key = jax.random.split(rng, 4)

            # Ask
            x, evo_state = strategy.ask(rng_gen, evo_state, es_params)

            # Player 1
            a1_state = a1_state._replace(
                params=param_reshaper.reshape(x),
            )
            a1_mem = agent1.batch_reset(a1_mem, False)

            ####################### PMAP ########################
            batched_rng = jnp.tile(rng_key, (num_devices, 1, 1))

            (
                fitness,
                other_fitness,
                traj_1,
                traj_2,
                a2_metrics,
                t1,
                t2,
                env_state,
            ) = pmapped_rollout(
                batched_rng,
                env,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
            )
            ####################### PMAP ########################
            fitness_re = fit_shaper.apply(x, fitness)  # Maximize fitness

            # Tell
            evo_state = strategy.tell(
                x, fitness_re - fitness_re.mean(), evo_state, es_params
            )

            # Logging
            log = es_logging.update(log, x, fitness)
            if log["gen_counter"] % 100 == 0:
                if self.algo == "OpenES" or self.algo == "PGPE":
                    jnp.save(
                        os.path.join(
                            self.save_dir,
                            f"mean_param_{log['gen_counter']}.npy",
                        ),
                        evo_state.mean,
                    )

                es_logging.save(
                    log,
                    os.path.join(
                        self.save_dir, f"generation_{log['gen_counter']}"
                    ),
                )

            if gen % log_interval == 0:
                if self.args.env_type == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(), self.cg_stats(env_state)
                    )
                    # print(traj_1.rewards.shape)
                    # (outer_steps, inner_steps, popsize, num_opps, num_envs )
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
                    "train/episode_reward/player_1": float(rewards_0.mean()),
                    "train/episode_reward/player_2": float(rewards_1.mean()),
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
                }
                # revert to work with Python 3.7
                # wandb_log = wandb_log | env_stats
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
                # TODO: Only works with 3.9 and i can't get 3.9 on colab
                # agent2._logger.metrics = (
                #     agent2._logger.metrics | flattened_metrics
                # )

                # agent1._logger.metrics = (
                #     agent1._logger.metrics | flattened_metrics
                # )
                agents.log(watchers)
                wandb.log(wandb_log)

        return agents