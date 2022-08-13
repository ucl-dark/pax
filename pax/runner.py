import time
from typing import List, NamedTuple

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

        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents.agents
        rng = jax.random.PRNGKey(0)

        if self.args.agent2 in ["Naive", "PPO", "PPO_memory"]:
            # let the RNG tell us the batch size.
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None, None), 0
            )
        elif self.args.agent2 == "NaiveEx":
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            # TODO: change fixed to take hidden
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )

        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy, 0, 0))
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        # batch MemoryState not TrainingState
        agent1.batch_init = jax.vmap(
            agent1.make_initial_state,
            (None, None, 0),
            (None, 0),
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # this needs to move into independent learners too
        init_hidden = jnp.tile(agent1._mem.hidden, (self.num_opps, 1, 1))
        a1_state, a1_mem = agent1.batch_init(
            rng, (env.observation_spec().num_values,), init_hidden
        )

        # run actual loop
        for i in range(
            0, max(int(num_episodes / (env.num_envs * self.num_opps)), 1)
        ):
            rng, _ = jax.random.split(rng)
            t_init, env_state = env.runner_reset((self.num_opps, env.num_envs))
            a1_mem = agent1.batch_reset(a1_mem, False)

            if self.args.agent2 in ["Naive", "PPO", "PPO_memory"]:
                if self.args.agent2 == "PPO_memory":
                    init_hidden = jnp.zeros((env.num_envs, agent2._gru_dim))
                else:
                    init_hidden = jnp.zeros((env.num_envs, 1))

                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps),
                    (env.observation_spec().num_values,),
                    init_hidden,
                )
            elif self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(t_init[1])
            else:
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps),
                    (env.observation_spec().num_values,),
                )

            # num trials
            vals, trajectories = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
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
                reduce_outer_traj(trajectories[0]),
                self.reduce_opp_dim(final_t1),
                a1_state,
                self.reduce_opp_dim(a1_mem),
            )
            self.train_episodes += 1
            rewards_0 = trajectories[0].rewards.mean()
            rewards_1 = trajectories[1].rewards.mean()

            if i % 5 == 0:
                print(
                    f"Total Episode Reward: {float(rewards_0.mean()), float(rewards_1.mean())}"
                )

                visits = self.state_visitation(trajectories[0])
                print(f"State Frequency: {visits}")

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
        agent1._mem = agent1.reset_memory(agent1._mem, True)
        agent2._mem = agent2.reset_memory(agent2._mem, True)

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
