from datetime import datetime
import os
import time
from typing import List, NamedTuple

from dm_env import TimeStep
from evosax import FitnessShaper, ParameterReshaper, OpenES, CMA_ES, PGPE
import jax
import jax.numpy as jnp
import wandb

# TODO: import when evosax library is updated
# from evosax.utils import ESLog
from pax.watchers import ESLog


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

        # TESTING CODE
        self.num_gens = args.num_gens
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

        # def _state_visitation(traj: Sample, final_t: TimeStep) -> List:
        #     # obs [num_outer_steps, num_inner_steps, num_opps, num_envs, ...]
        #     # final_t [num_opps, num_envs, ...]
        #     num_timesteps = (
        #         traj.observations.shape[0] * traj.observations.shape[1]
        #     )
        #     obs = jnp.reshape(
        #         traj.observations,
        #         (num_timesteps,) + traj.observations.shape[2:],
        #     )
        #     final_obs = jax.lax.expand_dims(final_t.observation, [0])
        #     obs = jnp.argmax(jnp.append(obs, final_obs, axis=0), axis=-1)
        #     return jnp.bincount(obs.flatten(), length=5)

        def _state_visitation(traj: Sample) -> List:
            obs = jnp.argmax(traj.observations, axis=-1)
            return jnp.bincount(obs.flatten(), length=5)

        self.state_visitation = jax.jit(_state_visitation)

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

        print("Training")
        print("-----------------------")

        agent1, agent2 = agents.agents
        num_opps = self.num_opps
        rng, _ = jax.random.split(self.random_key)

        # strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper

        # param_reshaper = ParameterReshaper(agent1._state.params)
        # TODO: Move these to agent_setup()
        if self.args.es.algo == "OpenES":
            print("Algo: OpenES")
            strategy = OpenES(
                num_dims=param_reshaper.total_params,
                popsize=self.popsize,
            )
            # Uncomment for specific param choices
            # Update basic parameters of PGPE strategy
            # es_params = strategy.default_params.replace(
            #     sigma_init=self.sigma_init,
            #     sigma_decay=self.sigma_decay,
            #     sigma_limit=self.sigma_limit,
            #     init_min=self.init_min,
            #     init_max=self.init_max,
            #     clip_min=self.clip_min,
            #     clip_max=self.clip_max,
            # )
            # # # Update optimizer-specific parameters of Adam
            # es_params = es_params.replace(
            #     opt_params=es_params.opt_params.replace(
            #         lrate_init=self.lrate_init,
            #         lrate_decay=self.lrate_decay,
            #         lrate_limit=self.lrate_limit,
            #         beta_1=self.beta_1,
            #         beta_2=self.beta_2,
            #         eps=self.eps,
            #     )
            # )
            # es_params = strategy.default_params
        elif self.algo == "CMA_ES":
            print("Algo: CMA_ES")
            strategy = CMA_ES(
                num_dims=param_reshaper.total_params,
                popsize=self.popsize,
                elite_ratio=0.5,
            )
            # Uncomment for default params
            es_params = strategy.default_params
        elif self.algo == "PGPE":
            print("Algo: PGPE")
            strategy = PGPE(
                num_dims=param_reshaper.total_params,
                popsize=self.popsize,
                opt_name="adam",
            )
            # Uncomment for default params
            es_params = strategy.default_params
        else:
            raise NotImplementedError
        evo_state = strategy.initialize(rng, es_params)
        # Instantiate jittable fitness shaper (e.g. for Open ES)
        fit_shaper = FitnessShaper(maximize=True)
        # Init logging
        es_logging = ESLog(
            param_reshaper.total_params,
            self.num_gens,
            top_k=self.top_k,
            maximize=True,
        )
        log = es_logging.initialize()
        # init agents
        init_hidden = jnp.tile(
            agent1._mem.hidden,
            (self.args.popsize, self.args.num_opps, 1, 1),
        )
        agent1._state, agent1._mem = agent1.batch_init(
            jax.random.split(agent1._state.random_key, self.args.popsize),
            init_hidden,
        )
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step),
        )
        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem
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

            # print("Training")
            # print("-----------------------")
            # # Initialize agents and RNG
            # agent1, agent2 = agents.agents
            # rng, _ = jax.random.split(self.random_key)

            # # Initialize evolution
            # num_gens = num_episodes
            # strategy = self.strategy
            # es_params = self.es_params
            # param_reshaper = self.param_reshaper
            # popsize = self.popsize
            # num_opps = self.num_opps
            # evo_state = strategy.initialize(rng, es_params)
            # fit_shaper = FitnessShaper(maximize=True)
            # es_logging = ESLog(
            #     param_reshaper.total_params,
            #     num_gens,
            #     top_k=self.top_k,
            #     maximize=True,
            # )
            # log = es_logging.initialize()

            # # Evolution specific: add pop size dimension
            # env.batch_step = jax.jit(
            #     jax.vmap(env.batch_step),
            # )

            # # Evolution specific: initialize player 1
            # init_hidden = jnp.tile(
            #     agent1._mem.hidden,
            #     (popsize, num_opps, 1, 1),
            # )
            # agent1._state, agent1._mem = agent1.batch_init(
            #     jax.random.split(agent1._state.random_key, popsize),
            #     init_hidden,
            # )

            # a1_state, a1_mem = agent1._state, agent1._mem
            # a2_state, a2_mem = agent2._state, agent2._mem

            # # num_gens
            # for gen in range(0, int(num_episodes)):
            #     rng, _ = jax.random.split(rng)
            #     #  rng, rng_run, rng_gen, rng_key = jax.random.split(rng, 4)
            #     t_init, env_state = env.runner_reset(
            #         (popsize, num_opps, env.num_envs), rng
            #     )
            #     # Prepare player 1
            #     rng, rng_gen, _ = jax.random.split(rng, 3)
            #     x, evo_state = strategy.ask(rng_gen, evo_state, es_params)

            #     a1_state = a1_state._replace(
            #         params=param_reshaper.reshape(x),
            #     )
            #     # print(a1_state.params)
            #     a1_mem = agent1.batch_reset(a1_mem, False)

            #     init_hidden = jnp.tile(
            #         agent2._mem.hidden, (popsize, num_opps, 1, 1)
            #     )

            #     # Prepare player 2
            #     a2_keys = jax.random.split(rng, popsize * num_opps).reshape(
            #         self.popsize, num_opps, -1
            #     )
            #     a2_state, a2_mem = agent2.batch_init(
            #         a2_keys,
            #         init_hidden,
            #     )

            #     a1, a1_state, a1_mem = agent1.batch_policy(
            #         a1_state,
            #         t_init[0].observation,
            #         a1_mem,
            #     )

            #     vals, trajectories = jax.lax.scan(
            #         _outer_rollout,
            #         (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
            #         None,
            #         length=env.num_trials,
            #     )

            #     final_t1 = vals[0]._replace(
            #         step_type=2 * jnp.ones_like(vals[0].step_type)
            #     )

            #     # calculate fitness
            #     fitness = trajectories[0].rewards.mean(axis=(0, 1, 3, 4))
            #     other_fitness = trajectories[1].rewards.mean(axis=(0, 1, 3, 4))

            #     # shape the fitness
            #     fitness_re = fit_shaper.apply(x, fitness)
            #     evo_state = strategy.tell(
            #         x, fitness_re - fitness_re.mean(), evo_state, es_params
            #     )

            #     visits = self.state_visitation(trajectories[0], final_t1)
            #     prob_visits = visits / visits.sum()

            # Logging
            log = es_logging.update(log, x, fitness)
            if log["gen_counter"] % 100 == 0:
                if self.algo == "OpenES" or self.algo == "PGPE":
                    jnp.save(
                        os.path.join(
                            self.save_dir,
                            f"mean_param_{self.generations}.npy",
                        ),
                        evo_state.mean,
                    )

                es_logging.save(
                    log,
                    os.path.join(
                        self.save_dir, f"generation_{self.generations}"
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
