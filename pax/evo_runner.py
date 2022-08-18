from datetime import datetime
from multiprocessing.sharedctypes import Value
import time
from typing import List, NamedTuple
import os

from evosax import OpenES, CMA_ES, PGPE, FitnessShaper, ParameterReshaper

# from evosax.utils import ESLog
from pax.utils import ESLog
import jax
import jax.numpy as jnp

import wandb


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

    def __init__(self, args, strategy, es_params, param_reshaper):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.algo = args.es.algo
        self.num_opps = args.num_opponents
        self.num_gens = args.num_gens
        self.popsize = args.popsize
        self.generations = 0
        self.top_k = args.top_k
        self.log_dir = f"{os.getcwd()}/pax/log/{str(datetime.now()).replace(' ', '_')}_{self.args.es.algo}"
        self.random_key = jax.random.PRNGKey(args.seed)
        self.strategy = strategy
        self.es_params = es_params
        self.param_reshaper = param_reshaper

        # OpenES hyperparameters
        self.sigma_init = args.es.sigma_init
        self.sigma_decay = args.es.sigma_decay
        self.sigma_limit = args.es.sigma_limit
        self.init_min = args.es.init_min
        self.init_max = args.es.init_max
        self.clip_min = args.es.clip_min
        self.clip_max = args.es.lrate_decay
        self.lrate_init = args.es.lrate_decay
        self.lrate_decay = args.es.lrate_decay
        self.lrate_limit = args.es.lrate_decay
        self.beta_1 = args.es.lrate_decay
        self.beta_2 = args.es.lrate_decay
        self.eps = args.es.lrate_decay

        def _reshape_opp_dim(x):
            # x: [num_opps, num_envs ...]
            # x: [batch_size, ...]
            batch_size = args.num_envs * args.num_opponents
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), x
            )

        def _state_visitation(traj: Sample) -> List:
            obs = jnp.argmax(traj.observations, axis=-1)
            return jnp.bincount(obs.flatten(), length=5)

        def _get_state_visitation(obs: jnp.ndarray) -> List:
            return jnp.bincount(obs.flatten(), length=5)

        def _get_observations(traj: Sample) -> List:
            # Shape: [outer_loop, inner_loop, popsize, num_opps, num_envs]
            obs = jnp.argmax(traj.observations, axis=-1)
            return obs

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        self.state_visitation = jax.jit(_state_visitation)
        self.get_state_visitation = jax.jit(_get_state_visitation)
        self.get_observations = jax.jit(_get_observations)

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper
        num_opps = self.num_opps
        agent1, agent2 = agents.agents

        # TODO: remove and put into experiment
        os.mkdir(self.log_dir)
        rng, _ = jax.random.split(self.random_key)
        evo_state = strategy.initialize(rng, es_params)
        fit_shaper = FitnessShaper(maximize=True)
        es_logging = ESLog(
            param_reshaper.total_params,
            self.num_gens,
            top_k=self.top_k,
            maximize=True,
        )
        log = es_logging.initialize()

        # batch environment step for evolution games (i.e. add popsize)
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step),
        )
        # init player 1
        init_hidden = jnp.tile(
            agent1._mem.hidden,
            (self.args.popsize, self.args.num_opponents, 1, 1),
        )
        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(agent1._state.random_key, self.args.popsize),
            init_hidden,
        )

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        def _inner_rollout(carry, unused):
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
            final_t2 = vals[1]._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state, a2_mem = agent2.batch_update(
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
            ), trajectories

        for gen in range(self.num_gens):
            rng, _ = jax.random.split(rng)
            t_init, env_state = env.runner_reset(
                (self.popsize, num_opps, env.num_envs), rng
            )

            rng, rng_gen, _ = jax.random.split(rng, 3)
            x, evo_state = strategy.ask(rng_gen, evo_state, es_params)

            a1_state = a1_state._replace(
                params=param_reshaper.reshape(x),
            )
            a1_mem = agent1.batch_reset(a1_mem, False)

            a2_keys = jax.random.split(rng, self.popsize * num_opps).reshape(
                self.popsize, num_opps, -1
            )
            a2_state, a2_mem = agent2.batch_init(
                a2_keys,
                init_hidden,
            )

            a1, a1_state, a1_new_mem = agent1.batch_policy(
                a1_state,
                t_init[0].observation,
                a1_mem,
            )

            # num trials
            _, trajectories = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            # calculate fitness
            fitness = trajectories[0].rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = trajectories[1].rewards.mean(axis=(0, 1, 3, 4))

            # shape the fitness
            fitness_re = fit_shaper.apply(x, fitness)
            evo_state = strategy.tell(
                x, fitness_re - fitness_re.mean(), evo_state, es_params
            )

            visits = self.state_visitation(trajectories[0])
            prob_visits = visits / visits.sum()

            # Logging
            log = es_logging.update(log, x, fitness)
            if log["gen_counter"] % 100 == 0:
                if self.algo == "OpenES" or self.algo == "PGPE":
                    jnp.save(
                        os.path.join(
                            self.log_dir,
                            f"mean_param_{self.generations}.npy",
                        ),
                        evo_state.mean,
                    )

                es_logging.save(
                    log,
                    os.path.join(
                        self.log_dir, f"generation_{self.generations}"
                    ),
                )

            print(f"Generation: {log['gen_counter']}")
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"Fitness: {fitness.mean()} | Other Fitness: {other_fitness.mean()}"
            )
            print(f"State Visitation: {prob_visits}")
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

            wandb_log = {
                "generations": self.generations,
                "train/fitness/player_1": float(fitness.mean()),
                "train/fitness/player_2": float(other_fitness.mean()),
                "train/fitness/top_overall_mean": log["log_top_mean"][gen],
                "train/fitness/top_overall_std": log["log_top_std"][gen],
                "train/fitness/top_gen_mean": log["log_top_gen_mean"][gen],
                "train/fitness/top_gen_std": log["log_top_gen_std"][gen],
                "train/fitness/gen_std": log["log_gen_std"][gen],
                "train/state_visitation/CC": prob_visits[0],
                "train/state_visitation/CD": prob_visits[1],
                "train/state_visitation/DC": prob_visits[2],
                "train/state_visitation/DD": prob_visits[3],
                "train/state_visitation/START": prob_visits[4],
                "train/time/minutes": float(
                    (time.time() - self.start_time) / 60
                ),
                "train/time/seconds": float((time.time() - self.start_time)),
            }

            for idx, (overall_fitness, gen_fitness) in enumerate(
                zip(log["top_fitness"], log["top_gen_fitness"])
            ):
                wandb_log[
                    f"train/fitness/top_overall_agent_{idx+1}"
                ] = overall_fitness
                wandb_log[f"train/fitness/top_gen_agent_{idx+1}"] = gen_fitness

            if watchers:
                agents.log(watchers)
                wandb.log(wandb_log)
        print()
        return agents
