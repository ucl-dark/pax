import time
from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp

import wandb
from pax.independent_learners import IndependentLearners
from pax.strategies import Defect, TitForTat, TrainingState
import numpy as np
from evosax import ParameterReshaper
from evosax.utils import ESLog


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
        self.generations = 0

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents

        def _inner_rollout(carry, unused):
            t1, t2, a1_state, a2_state, env_state = carry
            a1, new_a1_state = agent1._policy(
                a1_state.params, t1.observation, a1_state
            )
            a2, new_a2_state = agent2._policy(
                a2_state.params, t2.observation, a2_state
            )
            (tprime_1, tprime_2), env_state = env.runner_step(
                [
                    a1,
                    a2,
                ],
                env_state,
            )
            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                new_a1_state.extras["log_probs"],
                new_a1_state.extras["values"],
                tprime_1.last() * jnp.zeros(env.num_envs),
                a1_state.hidden,
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                new_a2_state.extras["log_probs"],
                new_a2_state.extras["values"],
                tprime_2.last(),
                a2_state.hiddens,
            )
            return (
                tprime_1,
                tprime_2,
                new_a1_state,
                new_a2_state,
                env_state,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            t1, t2, a1_state, a2_state, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                (t1, t2, a1_state, a2_state, env_state),
                None,
                length=env.inner_episode_length,
            )

            # do second agent update
            final_t2 = vals[1]._replace(step_type=2)
            a2_state = vals[3]
            a2_state = agent2.update(trajectories[1], final_t2, a2_state)
            return (t1, t2, a1_state, a2_state, env_state), trajectories

        rng = jax.random.PRNGKey(0)

        from evosax import CMA_ES

        param_reshaper = ParameterReshaper(agent1._state.params)

        popsize = 20
        strategy = CMA_ES(
            popsize=popsize,
            num_dims=param_reshaper.total_params,
            elite_ratio=0.5,
        )
        es_params = strategy.default_params
        evo_state = strategy.initialize(rng, es_params)

        es_logging = ESLog(
            param_reshaper.total_params,
            max(int(num_episodes / env.num_envs), 1),
            top_k=5,
            maximize=True,
        )
        log = es_logging.initialize()
        # run actual loop
        for generation in range(0, max(int(num_episodes / env.num_envs), 1)):
            t_init = env.reset()
            env_state = env.state
            rng, rng_gen, rng_eval = jax.random.split(rng, 3)
            x, evo_state = strategy.ask(rng_gen, evo_state, es_params)
            fitness = jnp.zeros((popsize))
            other_fitness = jnp.zeros((popsize))

            a1_state = agent1._state
            # map from numbers back into params
            for pop_idx in range(popsize):
                a1_state = a1_state._replace(
                    params=param_reshaper.reshape_single(x[pop_idx])
                )
                agent1._state = a1_state

                a1_state = agent1.reset_memory()
                a2_state = agent2.make_initial_state(
                    rng, (env.observation_spec().num_values,)
                )
                rng, _ = jax.random.split(rng)

                # num trials
                _, trajectories = jax.lax.scan(
                    _outer_rollout,
                    (*t_init, a1_state, a2_state, env_state),
                    None,
                    length=env.num_trials,
                )

                self.train_episodes += 1
                fitness = fitness.at[pop_idx].set(
                    (trajectories[0].rewards.mean())
                )
                other_fitness = other_fitness.at[pop_idx].set(
                    trajectories[1].rewards.mean()
                )
            log = es_logging.update(log, x, fitness)

            # update the evo
            evo_state = strategy.tell(x, -1 * fitness, evo_state, es_params)
            self.generations += 1
            print(
                f"Generation: {generation} | Best: {log['log_top_1'][generation]} | "
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
            # logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "Generation": self.generations,
                        "train/fitness/player_1": float(fitness.mean()),
                        "train/fitness/player_2": float(other_fitness.mean()),
                        "train/fitness/best_training_score": log["log_top_1"][
                            generation
                        ],
                        "train/fitness/top_agents_1": log["top_fitness"][0],
                        "train/fitness/top_agents_2": log["top_fitness"][1],
                        "train/fitness/top_agents_3": log["top_fitness"][2],
                        "train/fitness/top_agents_4": log["top_fitness"][3],
                        "train/fitness/top_agents_5": log["top_fitness"][4],
                    },
                )

        # update agents
        agents.agents[0]._state = param_reshaper.reshape(evo_state.best_member)
        agents.agents[1]._state = a2_state
        return agents

    def evaluate_loop(self, env, agents, num_episodes, watchers):
        """Run evaluation of agents against environment"""
        print("Evaluating")
        print("-----------------------")
        agents.eval(True)
        agent1, agent2 = agents.agents
        agent1.reset_memory()
        agent2.reset_memory()

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


if __name__ == "__main__":
    from pax.env import IteratedPrisonersDilemma

    agents = IndependentLearners([Defect(), TitForTat()])
    # TODO: accept the arguments from config file instead of hard-coding
    # Default to prisoner's dilemma
    env = IteratedPrisonersDilemma(50, 5)
    wandb.init(mode="online")

    def log_print(agent) -> dict:
        print(agent)
        return None
