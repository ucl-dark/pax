import time
from typing import List, NamedTuple

from evosax import OpenES, CMA_ES
from evosax import ParameterReshaper
from evosax.utils import ESLog
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


class Runner:
    """Holds the runner's state."""

    def __init__(self, args):
        self.train_steps = 0
        self.eval_steps = 0
        self.train_episodes = 0
        self.eval_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opponents
        self.num_gens = args.num_gens
        self.popsize = args.popsize
        self.generations = 0

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

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        self.state_visitation = jax.jit(_state_visitation)

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(0)
        param_reshaper = ParameterReshaper(agent1._state.params)

        strategy = OpenES(
            popsize=self.popsize,
            num_dims=param_reshaper.total_params,
        )

        # Uncomment for default params
        # es_params = strategy.default_params

        # Uncomment for specific param choices
        # Update basic parameters of PGPE strategy
        es_params = strategy.default_params.replace(
            sigma_init=self.sigma_init,
            sigma_decay=self.sigma_decay,
            sigma_limit=self.sigma_limit,
            init_min=self.init_min,
            init_max=self.init_max,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )

        # Update optimizer-specific parameters of Adam
        es_params = es_params.replace(
            opt_params=es_params.opt_params.replace(
                lrate_init=self.lrate_init,
                lrate_decay=self.lrate_decay,
                lrate_limit=self.lrate_limit,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                eps=self.eps,
            )
        )

        evo_state = strategy.initialize(rng, es_params)

        es_logging = ESLog(
            param_reshaper.total_params,
            self.num_gens,
            top_k=5,
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

        for _ in range(self.num_gens):
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
            self.train_episodes += 1
            fitness = trajectories[0].rewards.mean(axis=(0, 1, 3, 4))
            other_fitness = trajectories[1].rewards.mean(axis=(0, 1, 3, 4))

            # log
            self.train_episodes += 1
            print(
                f"Total Episode Reward: {float(fitness.mean()), float(other_fitness.mean())}"
            )
            visits = self.state_visitation(trajectories[0])
            print(f"State Frequency: {visits}")
            log = es_logging.update(log, x, fitness)

            # update the evo
            evo_state = strategy.tell(
                x, -1 * (fitness - fitness.mean()), evo_state, es_params
            )

            print(
                f"Generation: {self.generations} | Best: {log['log_top_1'][self.generations]} | "
                f"Fitness: {fitness.mean()} | Other Fitness: {other_fitness.mean()}"
            )
            print("----------")
            print("Top Agents")
            print("----------")
            print(f"Agent {1} | Fitness: {log['top_fitness'][0]}")
            print(f"Agent {2} | Fitness: {log['top_fitness'][1]}")
            print(f"Agent {3} | Fitness: {log['top_fitness'][2]}")
            print(f"Agent {4} | Fitness: {log['top_fitness'][3]}")
            print(f"Agent {5} | Fitness: {log['top_fitness'][4]}")
            # for idx, agent_fitness in enumerate(log['top_fitness']):
            #     print(f"Agent {idx} | Fitness: {agent_fitness}")
            print()
            self.generations += 1

            # logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "episodes": self.train_episodes,
                        "train/episode_reward/player_1": float(fitness.mean()),
                        "train/episode_reward/player_2": float(
                            other_fitness.mean()
                        ),
                        "wallclock": float(time.time.now() - self.start_time),
                    },
                )
        print()
        # update agents
        agents.agents[0]._state = a1_state
        agents.agents[1]._state = a2_state
        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        agent1, agent2 = agents.agents
        agent1._mem = agent1.reset_memory(agent1._mem, eval=True)
        agent2._mem = agent2.reset_memory(agent2._mem)

        for _ in range(num_episodes // env.num_envs):
            rewards_0, rewards_1 = [], []
            timesteps = env.reset()
            for _ in range(env.episode_length):
                actions = agents.select_action(timesteps)
                timesteps = env.step(actions)
                r_0, r_1 = timesteps[0].reward, timesteps[1].reward
                rewards_0.append(timesteps[0].reward)
                rewards_1.append(timesteps[1].reward)

                # update evaluation steps
                self.eval_steps += 1

                if watchers:
                    agents.log(watchers)
                    env_info = {
                        "eval/evaluation_steps": self.eval_steps,
                        "eval/step_reward/player_1": float(
                            jnp.array(r_0).mean()
                        ),
                        "eval/step_reward/player_2": float(
                            jnp.array(r_1).mean()
                        ),
                    }
                    wandb.log(env_info)

            # end of episode stats
            self.eval_episodes += 1
            rewards_0 = jnp.array(rewards_0)
            rewards_1 = jnp.array(rewards_1)

            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )
            if watchers:
                wandb.log(
                    {
                        "train/episodes": self.train_episodes,
                        "eval/episodes": self.eval_episodes,
                        "eval/episode_reward/player_1": float(
                            rewards_0.mean()
                        ),
                        "eval/episode_reward/player_2": float(
                            rewards_1.mean()
                        ),
                        "eval/joint_reward": float(
                            rewards_0.mean() + rewards_1.mean() * 0.5
                        ),
                    }
                )
        agents.eval(False)
        print()
        return agents
