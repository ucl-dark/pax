from datetime import datetime
import time
from typing import List, NamedTuple
import os

from evosax import OpenES, CMA_ES, PGPE, FitnessShaper
from evosax import ParameterReshaper

# from evosax.utils import ESLog
from pax.utils import ESLog
import jax
import jax.numpy as jnp


import wandb

path = os.getcwd()
print(path)


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class Runner:
    """Holds the runner's state."""

    def __init__(self, args):
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
        if self.args.only_eval:
            print("Skipping Training: only_eval = True")
            print()
            return agents
        print("Training")
        print("-----------------------")
        os.mkdir(self.log_dir)
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(0)
        param_reshaper = ParameterReshaper(agent1._state.params)

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

            # # Update optimizer-specific parameters of Adam
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
            es_params = strategy.default_params
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
            # top_k=self.popsize,
            maximize=True,
        )

        log = es_logging.initialize()

        # vmap pop_size, num_opps dims
        num_opps = self.num_opps
        env.batch_step = jax.jit(
            jax.vmap(
                jax.vmap(env.runner_step, (0, None), (0, None)),
                (0, None),
                (0, None),
            )
        )

        # batch over TimeStep, TrainingState
        agent2.make_initial_state = jax.jit(
            jax.vmap(
                jax.vmap(agent2.make_initial_state, (0, None), 0), (0, None), 0
            )
        )

        agent2.batch_policy = jax.jit(jax.vmap(jax.vmap(agent2._policy)))
        # traj_batch : [inner_loop, pop_size, num_opps, num_envs, ...]
        agent2.batch_update = jax.jit(
            jax.vmap(
                jax.vmap(agent2.update, (1, 0, 0, 0), (0, 0)),
                (1, 0, 0, 0),
                (0, 0),
            )
        )

        # batch over TimeStep, TrainingState
        agent1.make_initial_state = jax.vmap(
            jax.vmap(agent1.make_initial_state, (None, None, 0), (None, 0)),
            (0, None, 0),
            (0, 0),
        )

        agent1.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent1.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent2.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent2.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(
                jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0)),
                0,
                0,
            )
        )
        a1_state, a1_mem = agent1.make_initial_state(
            jax.random.split(rng, self.popsize),
            (env.observation_spec().num_values,),
            jnp.zeros((self.popsize, num_opps, env.num_envs, agent1._gru_dim)),
        )

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

            # do second agent update
            final_t2 = vals[1]._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state = vals[4]
            a2_mem = vals[5]
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
                (self.popsize, num_opps, env.num_envs)
            )

            rng, rng_gen, _ = jax.random.split(rng, 3)
            x, evo_state = strategy.ask(rng_gen, evo_state, es_params)

            batched_params = param_reshaper.reshape(x)
            a1_state = a1_state._replace(params=batched_params)
            a1_mem = agent1.batch_reset(a1_mem, False)

            a2_state, a2_mem = agent2.make_initial_state(
                jax.random.split(rng, self.popsize * num_opps).reshape(
                    self.popsize, num_opps, -1
                ),
                (env.observation_spec().num_values,),
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
            if self.generations % 1 == 0:
                if self.algo == "OpenES" or self.algo == "PGPE":
                    jnp.save(
                        os.path.join(
                            self.log_dir,
                            f"mean_param_{log['gen_counter']}.npy",
                        ),
                        evo_state.mean,
                    )

                es_logging.save(
                    log,
                    os.path.join(
                        self.log_dir, f"generation_{log['gen_counter']}"
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

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Evaluating")
        print("-----------------------")
        eval_dir = f"{os.getcwd()}/pax/log/"
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(0)
        param_reshaper = ParameterReshaper(agent1._state.params)
        es_logging = ESLog(
            param_reshaper.total_params,
            self.num_gens,
            top_k=self.top_k,
            maximize=True,
        )

        # Evaluation parameters
        num_opps = 20
        eval_popsize = 1

        # TODO: Remove after f/batched_naive is merged. Moved to IndependentLearners
        env.batch_step = jax.jit(
            jax.vmap(
                jax.vmap(env.runner_step, (0, None), (0, None)),
                (0, None),
                (0, None),
            )
        )

        # Use only when you want to evaluate; you need to jit and vmap all of the functions first.
        # TODO: Will be moved to indepenedent learners in upcoming PR
        if self.args.only_eval:
            # batch over TimeStep, TrainingState
            agent2.make_initial_state = jax.jit(
                jax.vmap(
                    jax.vmap(agent2.make_initial_state, (0, None), 0),
                    (0, None),
                    0,
                )
            )

            agent2.batch_policy = jax.jit(jax.vmap(jax.vmap(agent2._policy)))
            # traj_batch : [inner_loop, pop_size, num_opps, num_envs, ...]
            agent2.batch_update = jax.jit(
                jax.vmap(
                    jax.vmap(agent2.update, (1, 0, 0, 0), (0, 0)),
                    (1, 0, 0, 0),
                    (0, 0),
                )
            )

            # batch over TimeStep, TrainingState
            agent1.make_initial_state = jax.vmap(
                jax.vmap(
                    agent1.make_initial_state, (None, None, 0), (None, 0)
                ),
                (0, None, 0),
                (0, 0),
            )

            agent1.batch_reset = jax.jit(
                jax.vmap(
                    jax.vmap(agent1.reset_memory, (0, None), 0), (0, None), 0
                ),
                static_argnums=1,
            )

            agent2.batch_reset = jax.jit(
                jax.vmap(
                    jax.vmap(agent2.reset_memory, (0, None), 0), (0, None), 0
                ),
                static_argnums=1,
            )

            agent1.batch_policy = jax.jit(
                jax.vmap(
                    jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0)),
                    0,
                    0,
                )
            )

            # Load previous parameters
            log = es_logging.load(os.path.join(eval_dir, self.args.model_path))

        else:
            # Load previous parameters from the training run above
            log = es_logging.load(
                os.path.join(self.log_dir, f"generation_{self.num_gens}")
            )

        a1_state, a1_mem = agent1.make_initial_state(
            jax.random.split(rng, eval_popsize),
            (env.observation_spec().num_values,),
            jnp.zeros((eval_popsize, num_opps, env.num_envs, agent1._gru_dim)),
        )

        # TODO: Add the ability to evaluate every n number of generations
        #     log = es_logging.load(
        #     os.path.join(self.log_dir, f"generation_{self.generations}")
        # )

        params = param_reshaper.reshape(log["top_gen_params"][0:1])
        params = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, eval_popsize, axis=0), params
        )
        a1_state = a1_state._replace(params=params)

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

            # do second agent update
            final_t2 = vals[1]._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state = vals[4]
            a2_mem = vals[5]
            a2_state, a2_mem = agent2.batch_update(
                trajectories[1], final_t2, a2_state, a2_mem
            )
            # reset player 2 mem to num_envs = 1
            a2_mem = agent2.batch_reset(a2_mem, True)

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
            ), trajectories

        for _ in range(1):
            rng, _ = jax.random.split(rng)
            t_init, env_state = env.runner_reset(
                (eval_popsize, num_opps, env.num_envs)
            )

            a2_state, a2_mem = agent2.make_initial_state(
                jax.random.split(rng, eval_popsize * num_opps).reshape(
                    eval_popsize, num_opps, -1
                ),
                (env.observation_spec().num_values,),
            )

            # EVAL ONLY
            a1_mem = agent1.batch_reset(a1_mem, True)
            a2_mem = agent2.batch_reset(a2_mem, True)

            _, trajectories = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            # calculate fitness
            fitness = trajectories[0].rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = trajectories[1].rewards.mean(axis=(0, 1, 3, 4))

            # calculate state visitation
            visits = self.state_visitation(trajectories[0])
            prob_visits = visits / visits.sum()

            print(f"Summary | Number of opponents: {num_opps}")
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                f"EARL: {fitness.mean()} | Naive Learners: {other_fitness.mean()}"
            )
            # print(f"State Frequency: {visits}")
            print(f"State Visitation: {prob_visits}")
            print(
                "--------------------------------------------------------------------------"
            )

            # Logging step rewards loop
            # Shape: [outer_loop, inner_loop, popsize, num_opps, num_envs]
            observations = self.get_observations(trajectories[0])
            p1_rewards = trajectories[0].rewards
            p2_rewards = trajectories[1].rewards
            for opp_i in range(num_opps):
                # TODO: Figure out a way to allow for popsize=1 so you can remove 0
                # Canonically player 1
                obs_opp_i = observations[:, :, :, opp_i, :].squeeze()
                p1_rew_opp_i = p1_rewards[:, :, :, opp_i, :].squeeze()
                p2_rew_opp_i = p2_rewards[:, :, :, opp_i, :].squeeze()
                inner_steps = 0
                for out_step in range(env.num_trials):
                    obs_outer_opp_i = obs_opp_i[out_step]
                    p1_ep_rew_opp_i = p1_rew_opp_i[out_step]
                    p2_ep_rew_opp_i = p2_rew_opp_i[out_step]
                    p1_ep_mean_rew_opp_i = p1_rew_opp_i[out_step].mean()
                    p2_ep_mean_rew_opp_i = p2_rew_opp_i[out_step].mean()
                    state_visit_trial_i = self.get_state_visitation(
                        obs_outer_opp_i
                    )
                    prob_visits = (
                        state_visit_trial_i / state_visit_trial_i.sum()
                    )
                    if watchers:
                        wandb.log(
                            {
                                "eval/trial": out_step + 1,
                                f"eval/reward_trial/player_1_opp_{opp_i+1}": p1_ep_mean_rew_opp_i,
                                f"eval/reward_trial/player_2_opp_{opp_i+1}": p2_ep_mean_rew_opp_i,
                                f"eval/state_visitation_trial/CC_opp_{opp_i+1}": state_visit_trial_i[
                                    0
                                ],
                                f"eval/state_visitation_trial/CD_opp_{opp_i+1}": state_visit_trial_i[
                                    1
                                ],
                                f"eval/state_visitation_trial/DC_opp_{opp_i+1}": state_visit_trial_i[
                                    2
                                ],
                                f"eval/state_visitation_trial/DD_opp_{opp_i+1}": state_visit_trial_i[
                                    3
                                ],
                                f"eval/state_visitation_trial/START_opp{opp_i+1}": state_visit_trial_i[
                                    4
                                ],
                                f"eval/state_visitation_trial/CC_probability_opp_{opp_i+1}": prob_visits[
                                    0
                                ],
                                f"eval/state_visitation_trial/CD_probability_opp_{opp_i+1}": prob_visits[
                                    1
                                ],
                                f"eval/state_visitation_trial/DC_probability_opp_{opp_i+1}": prob_visits[
                                    2
                                ],
                                f"eval/state_visitation_trial/DD_probability_opp_{opp_i+1}": prob_visits[
                                    3
                                ],
                                f"eval/state_visitation_trial/START_probability_opp_{opp_i+1}": prob_visits[
                                    4
                                ],
                            }
                        )

                    for in_step in range(env.inner_episode_length):
                        p1_step_rew_opp_i = p1_ep_rew_opp_i[in_step]
                        p2_step_rew_opp_i = p2_ep_rew_opp_i[in_step]
                        if watchers:
                            wandb.log(
                                {
                                    "eval/timestep": inner_steps + 1,
                                    f"eval/reward_step/player_1_opp_{opp_i+1}": p1_step_rew_opp_i,
                                    f"eval/reward_step/player_2_opp_{opp_i+1}": p2_step_rew_opp_i,
                                }
                            )
                        inner_steps += 1

                state_visit_opp_i = self.get_state_visitation(obs_opp_i)
                prob_visits = state_visit_opp_i / state_visit_opp_i.sum()
                print(
                    f"Opponent: {opp_i+1} | EARL: {p1_rew_opp_i.mean()} "
                    f"| Naive Learner {opp_i+1}: {p2_rew_opp_i.mean()}"
                )
                print(f"State visitations: {prob_visits}")
                if watchers:
                    wandb.log(
                        {
                            "eval/seeds": opp_i + 1,
                            "eval/reward_opp/player_1": float(
                                p1_rew_opp_i.mean()
                            ),
                            "eval/reward_opp/player_2_opp": float(
                                p2_rew_opp_i.mean()
                            ),
                            "eval/state_visitation_opp/CC": state_visit_opp_i[
                                0
                            ],
                            "eval/state_visitation_opp/CD": state_visit_opp_i[
                                1
                            ],
                            "eval/state_visitation_opp/DC": state_visit_opp_i[
                                2
                            ],
                            "eval/state_visitation_opp/DD": state_visit_opp_i[
                                3
                            ],
                            "eval/state_visitation_opp/START": state_visit_opp_i[
                                4
                            ],
                            "eval/state_visitation_opp/CC_probability": prob_visits[
                                0
                            ],
                            "eval/state_visitation_opp/CD_probability": prob_visits[
                                1
                            ],
                            "eval/state_visitation_opp/DC_probability": prob_visits[
                                2
                            ],
                            "eval/state_visitation_opp/DD_probability": prob_visits[
                                3
                            ],
                            "eval/state_visitation_opp/START_probability": prob_visits[
                                4
                            ],
                            "eval/time/minutes": float(
                                (time.time() - self.start_time) / 60
                            ),
                            "eval/time/seconds": float(
                                (time.time() - self.start_time)
                            ),
                        }
                    )

            if watchers:
                agents.log(watchers)
        print()
        return agents
