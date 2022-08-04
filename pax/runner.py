import time
from typing import List, NamedTuple

import jax
import jax.numpy as jnp

import wandb
from pax.independent_learners import IndependentLearners
from pax.ppo.ppo_gru import MemoryState
from pax.strategies import Defect, TitForTat


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
def reshape_meta_traj(traj: Sample) -> Sample:
    batch_size = traj.observations.shape[0] * traj.observations.shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), traj
    )


@jax.jit
def reshape_nl_traj(traj: Sample) -> Sample:
    batch_size = traj.observations.shape[1] * traj.observations.shape[2]
    return jax.tree_util.tree_map(
        lambda x: x.reshape(x.shape[:1] + (batch_size,) + x.shape[3:]), traj
    )


@jax.jit
def reshape_nl_step(mem: MemoryState) -> Sample:
    batch_size = mem.extras["values"].shape[0] * mem.extras["values"].shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), mem
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
        self.num_nl = args.num_opponents

    def train_loop(self, env, agents, num_episodes, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(0)

        # vmap once to support multiple naive learners
        num_nl = 200

        env.batch_step = jax.vmap(
            env.runner_step, (0, 0, (None, None)), ((0, 0), None)
        )

        agent2.batch_policy = jax.vmap(agent2._policy, (0, 0, 0), (0, 0))
        agent2.batch_update = jax.vmap(agent2.update, (1, 0, 0), 0)
        agent2.make_initial_state = jax.vmap(
            agent2.make_initial_state, (0, None), 0
        )

        # batch MemoryState not TrainingState
        agent1.reset_memory = jax.vmap(agent1.reset_memory, (0, None), 0)
        agent1.make_initial_state = jax.vmap(
            agent1.make_initial_state,
            in_axes=(None, None, 0),
            out_axes=(None, 0),
        )

        # this should batch over observations / training states but not params
        agent1.batch_policy = jax.vmap(
            agent1._policy, (None, 0, 0), (0, None, 0)
        )

        # on meta step we've added 2 time dims so NL dim is 0 -> 2
        agent1.batch_update = jax.vmap(agent1.update, (1, 0, None, 0), None)

        # this will come from the GE
        # a1_state = agent1.make_initial_state(
        #     jax.random.split(rng, num_nl),
        #     (env.observation_spec().num_values,),
        #     jnp.zeros((num_nl, env.num_envs, agent1._gru_dim)),
        # )
        a1_state, a1_mem = agent1.make_initial_state(
            rng,
            (env.observation_spec().num_values,),
            jnp.zeros((num_nl, env.num_envs, agent1._gru_dim)),
        )

        @jax.jit
        def reshape_nl_traj(traj: Sample) -> Sample:
            batch_size = env.num_envs * num_nl
            return jax.tree_util.tree_map(
                lambda x: x.reshape(x.shape[:1] + (batch_size,) + x.shape[3:]),
                traj,
            )

        @jax.jit
        def reshape_nl_step(mem: MemoryState) -> Sample:
            batch_size = env.num_envs * num_nl
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), mem
            )

        def _inner_rollout(carry, unused):
            t1, t2, a1_state, a1_mem, a2_state, env_state = carry

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                t1.observation,
                a1_mem,
            )

            a2, new_a2_state = agent2.batch_policy(
                a2_state.params, t2.observation, a2_state
            )

            (tprime_1, tprime_2), env_state = env.batch_step(
                a1,
                a2,
                env_state,
            )

            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                tprime_1.last() * jnp.zeros(env.num_envs),
                a1_mem.hidden,
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
                a1_state,
                a1_mem,
                new_a2_state,
                env_state,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            t1, t2, a1_state, a1_mem, a2_state, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                (t1, t2, a1_state, a1_mem, a2_state, env_state),
                None,
                length=env.inner_episode_length,
            )

            # do second agent update
            final_t2 = vals[1]._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state = vals[4]
            a2_state = agent2.batch_update(trajectories[1], final_t2, a2_state)

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                env_state,
            ), trajectories

        # run actual loop
        for _ in range(0, max(int(num_episodes / env.num_envs), 1)):
            t_init, env_state = env.runner_reset((num_nl, env.num_envs))
            # env_state = env.state
            rng, _ = jax.random.split(rng)

            # uniquely nl code required here.
            a1_mem = agent1.reset_memory(a1_mem, False)
            a2_state = agent2.make_initial_state(
                jax.random.split(rng, num_nl),
                (env.observation_spec().num_values,),
            )

            # num trials
            vals, trajectories = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, env_state),
                None,
                length=env.num_trials,
            )
            # update outer agent
            final_t1 = vals[0]._replace(
                step_type=2 * jnp.ones_like(vals[0].step_type)
            )
            a1_state = vals[2]
            a1_mem = vals[3]

            a1_state, _ = agent1.update(
                reshape_nl_traj(reshape_meta_traj(trajectories[0])),
                reshape_nl_step(final_t1),
                a1_state,
                reshape_nl_step(a1_mem),
            )

            self.train_episodes += 1
            rewards_0 = trajectories[0].rewards.mean()
            rewards_1 = trajectories[1].rewards.mean()
            print(
                f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
            )

            # logging
            if watchers:
                agents.log(watchers)
                wandb.log(
                    {
                        "episodes": self.train_episodes,
                        "train/episode_reward/player_1": float(
                            rewards_0.mean()
                        ),
                        "train/episode_reward/player_2": float(
                            rewards_1.mean()
                        ),
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
        agent1._mem = agent1.reset_memory(agent1._mem)
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
