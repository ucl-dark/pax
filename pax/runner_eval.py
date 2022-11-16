import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

import wandb
from pax.utils import load
from pax.watchers import cg_visitation, ipd_visitation

MAX_WANDB_CALLS = 10000


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class EvalRunner:
    """
    Evaluation runner provides a convenient example for quickly writing
    a shaping eval runner for PAX. The EvalRunner class can be used to
    run any two agents together either in a meta-game or regular game, it composes together agents,
    watchers, and the environment. Within the init, we declare vmaps and pmaps for training.
    Args:
        agents (Tuple[agents]):
            The set of agents that will run in the experiment. Note, ordering is important for
            logic used in the class.
        env (gymnax.envs.Environment):
            The environment that the agents will run in.
        args (NamedTuple):
            A tuple of experiment arguments used (usually provided by HydraConfig).
    """

    def __init__(self, agents, env, args):
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opps
        self.random_key = jax.random.PRNGKey(args.seed)
        self.run_path = args.run_path
        self.model_path = args.model_path
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)
        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )

        # VMAP for num opps: we vmap over the rng but not params
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        agent1, agent2 = agents

        if args.agent1 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent1.batch_init = jax.jit(jax.vmap(agent1.make_initial_state))
        else:
            # batch MemoryState not TrainingState
            agent1.batch_init = jax.vmap(
                agent1.make_initial_state,
                (None, 0),
                (None, 0),
            )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )

        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # batch all for Agent2
        if args.agent2 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )

        if args.agent2 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
            agent2._state, agent2._mem = agent2.batch_init(
                jax.random.split(agent2._state.random_key, args.num_opps),
                init_hidden,
            )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                obs1,
                a1_mem,
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                obs2,
                a2_mem,
            )
            (next_obs1, next_obs2), env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                (a1, a2),
                env_params,
            )

            traj1 = Sample(
                obs1,
                a1,
                rewards[0],
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )
            traj2 = Sample(
                obs2,
                a2,
                rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
            )
            return (
                rngs,
                next_obs1,
                next_obs2,
                rewards[0],
                rewards[1],
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
                env_params,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
            )
            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                a1_mem = agent1.meta_policy(a1_mem)

            # update second agent
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1],
                obs2,
                a2_state,
                a2_mem,
            )
            return (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ), (*trajectories, a2_metrics)

        self.rollout = jax.jit(_outer_rollout)

    def run_loop(self, env, env_params, agents, num_episodes, watchers):
        """Run evaluation of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        if watchers:
            wandb.restore(
                name=self.model_path, run_path=self.run_path, root=os.getcwd()
            )
        pretrained_params = load(self.model_path)
        a1_state = a1_state._replace(params=pretrained_params)

        num_iters = max(
            int(num_episodes / (self.args.num_envs * self.args.num_opps)), 1
        )
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Log Interval {log_interval}")

        # RNG are the same for num_opps but different for num_envs
        rngs = jnp.concatenate(
            [jax.random.split(rng, self.args.num_envs)] * self.args.num_opps
        ).reshape((self.args.num_opps, self.args.num_envs, -1))
        # run actual loop
        for i in range(num_episodes):
            obs, env_state = env.reset(rngs, env_params)
            rewards = [
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
                jnp.zeros((self.args.num_opps, self.args.num_envs)),
            ]

            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(obs[1])
            elif self.args.env_type in ["meta"]:
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps), a2_mem.hidden
                )
            # run trials
            vals, stack = jax.lax.scan(
                self.rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    a1_state,
                    a1_mem,
                    a2_state,
                    a2_mem,
                    env_state,
                    env_params,
                ),
                None,
                length=self.args.num_steps // self.args.num_inner_steps,
            )

            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
                env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # reset second agent memory
            a2_mem = agent2.batch_reset(a2_mem, False)

            # logging
            self.train_episodes += 1
            if i % log_interval == 0:
                print(f"Episode {i}")
                if self.args.env_id == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.cg_stats(env_state),
                    )
                    rewards_1 = traj_1.rewards.sum(axis=1).mean()
                    rewards_2 = traj_2.rewards.sum(axis=1).mean()

                elif self.args.env_type in [
                    "meta",
                    "sequential",
                ]:
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.ipd_stats(
                            traj_1.observations,
                            traj_1.actions,
                            obs1,
                        ),
                    )
                    rewards_1 = traj_1.rewards.mean()
                    rewards_2 = traj_2.rewards.mean()

                else:
                    rewards_1 = traj_1.rewards.mean()
                    rewards_2 = traj_2.rewards.mean()
                    env_stats = {}

                print(f"Env Stats: {env_stats}")
                print(
                    f"Total Episode Reward: {float(rewards_1.mean()), float(rewards_2.mean())}"
                )
                print()

                if watchers:
                    # metrics [outer_timesteps, num_opps]
                    flattened_metrics = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flattened_metrics
                    )

                    for watcher, agent in zip(watchers, agents):
                        watcher(agent)
                    wandb.log(
                        {
                            "episodes": self.train_episodes,
                            "train/episode_reward/player_1": float(
                                rewards_1.mean()
                            ),
                            "train/episode_reward/player_2": float(
                                rewards_2.mean()
                            ),
                        }
                        | env_stats,
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
