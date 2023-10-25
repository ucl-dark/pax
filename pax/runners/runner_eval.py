import os
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp

import wandb
from pax.utils import load
from pax.watchers import cg_visitation, ipd_visitation
from pax.watchers.c_rice import c_rice_eval_stats, c_rice_stats
from pax.watchers.fishery import fishery_stats
from pax.watchers.rice import rice_eval_stats, rice_stats

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
    watchers, and the environment. Within the init, we declare vmaps for training.
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
        self.run_path2 = args.get("run_path2", None)
        self.model_path2 = args.get("model_path2", None)
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

        agent1, agent2 = agents[0], agents[1]

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
                agent_order,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            rngs = rngs[:, :, 3, :]

            a1_actions = []
            new_a1_memories = []
            for _obs, _mem in zip(obs1, a1_mem):
                a1_action, a1_state, new_a1_memory = agent1.batch_policy(
                    a1_state,
                    _obs,
                    _mem,
                )
                a1_actions.append(a1_action)
                new_a1_memories.append(new_a1_memory)

            a2_actions = []
            new_a2_memories = []
            for _obs, _mem in zip(obs2, a2_mem):
                a2_action, a2_state, new_a2_memory = agent2.batch_policy(
                    a2_state,
                    _obs,
                    _mem,
                )
                a2_actions.append(a2_action)
                new_a2_memories.append(new_a2_memory)

            actions = jnp.asarray([*a1_actions, *a2_actions])[agent_order]
            obs, env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                tuple(actions),
                env_params,
            )

            inv_agent_order = jnp.argsort(agent_order)
            obs = jnp.asarray(obs)[inv_agent_order]
            rewards = jnp.asarray(rewards)[inv_agent_order]

            a1_trajectories = [
                Sample(
                    observation,
                    action,
                    reward * jnp.logical_not(done),
                    new_memory.extras["log_probs"],
                    new_memory.extras["values"],
                    done,
                    memory.hidden,
                )
                for observation, action, reward, new_memory, memory in zip(
                    obs1,
                    a2_actions,
                    rewards[: self.args.agent1_roles],
                    new_a1_memories,
                    a1_mem,
                )
            ]
            a2_trajectories = [
                Sample(
                    observation,
                    action,
                    reward * jnp.logical_not(done),
                    new_memory.extras["log_probs"],
                    new_memory.extras["values"],
                    done,
                    memory.hidden,
                )
                for observation, action, reward, new_memory, memory in zip(
                    obs2,
                    a2_actions,
                    rewards[self.args.agent1_roles :],
                    new_a2_memories,
                    a2_mem,
                )
            ]

            return (
                rngs,
                tuple(obs[: self.args.agent1_roles]),
                tuple(obs[self.args.agent1_roles :]),
                tuple(rewards[: self.args.agent1_roles]),
                tuple(rewards[self.args.agent1_roles :]),
                a1_state,
                tuple(new_a1_memories),
                a2_state,
                tuple(new_a2_memories),
                env_state,
                env_params,
                agent_order,
            ), (
                a1_trajectories,
                a2_trajectories,
                env_state,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, stack = jax.lax.scan(
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
                agent_order,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                a1_mem = agent1.meta_policy(a1_mem)

            # update second agent
            if self.args.agent2_learning is False:
                new_a2_memories = a2_mem
                a2_metrics = {}
            else:
                new_a2_memories = []
                for _obs, mem, traj in zip(obs2, a2_mem, stack[1]):
                    a2_state, a2_mem, a2_metrics = agent2.batch_update(
                        traj,
                        _obs,
                        a2_state,
                        mem,
                    )
                    new_a2_memories.append(a2_mem)
                new_a2_memories = tuple(new_a2_memories)
            return (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                a1_mem,
                a2_state,
                new_a2_memories,
                env_state,
                env_params,
                agent_order,
            ), (*stack, a2_metrics)

        self.rollout = jax.jit(_outer_rollout)

    def run_loop(self, env, env_params, agents, num_episodes, watchers):
        """Run evaluation of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents[0], agents[1]
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        preload_agent_2 = (
            self.model_path2 is not None and self.run_path2 is not None
        )

        if watchers and self.args.wandb.mode not in [
            "offline",
            "disabled",
        ]:
            wandb.restore(
                name=self.model_path, run_path=self.run_path, root=os.getcwd()
            )
            if preload_agent_2:
                wandb.restore(
                    name=self.model_path2,
                    run_path=self.run_path2,
                    root=os.getcwd(),
                )

        pretrained_params = load(self.model_path)
        a1_state = a1_state._replace(params=pretrained_params)

        if preload_agent_2:
            pretrained_params = load(self.model_path2)
            a2_state = a2_state._replace(params=pretrained_params)
            a2_pretrained_params = pretrained_params

        num_iters = max(
            int(num_episodes / (self.args.num_envs * self.args.num_opps)), 1
        )
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Log Interval {log_interval}")

        # RNG are the same for num_opps but different for num_envs

        a1_mem = (a1_mem,) * self.args.agent1_roles
        # run actual loop
        for i in range(num_episodes):
            rng, _ = jax.random.split(rng)
            rngs = jnp.concatenate(
                [jax.random.split(rng, self.args.num_envs)]
                * self.args.num_opps
            ).reshape((self.args.num_opps, self.args.num_envs, -1))

            obs, env_state = env.reset(rngs, env_params)
            rewards = [jnp.zeros((self.args.num_opps, self.args.num_envs))] * (
                self.args.agent1_roles + self.args.agent2_roles
            )

            if i % self.args.agent2_reset_interval == 0:
                if self.args.agent2 == "NaiveEx":
                    a2_state, a2_mem = agent2.batch_init(obs[1])
                elif self.args.env_type in ["meta"]:
                    # meta-experiments - init 2nd agent per trial
                    a2_state, a2_mem = agent2.batch_init(
                        jax.random.split(rng, self.num_opps), a2_mem.hidden
                    )

                if preload_agent_2:
                    # If we are preloading agent 2 we want to keep the state
                    a2_state = a2_state._replace(
                        params=jax.tree_util.tree_map(
                            lambda x: jnp.expand_dims(x, 0),
                            a2_pretrained_params,
                        )
                    )
                a2_mem = (a2_mem,) * self.args.agent2_roles

            agent_order = jnp.arange(self.args.num_players)
            if self.args.shuffle_players:
                agent_order = jax.random.permutation(rng, agent_order)

            # run trials
            vals, stack = jax.lax.scan(
                self.rollout,
                (
                    rngs,
                    tuple(obs[: self.args.agent1_roles]),
                    tuple(obs[self.args.agent1_roles :]),
                    tuple(rewards[: self.args.agent1_roles]),
                    tuple(rewards[self.args.agent1_roles :]),
                    a1_state,
                    a1_mem,
                    a2_state,
                    a2_mem,
                    env_state,
                    env_params,
                    agent_order,
                ),
                None,
                length=self.args.num_steps,
            )

            (
                rngs,
                obs1,
                obs2,
                r1,
                r2,
                a1_state,
                # We rename the memory since we do not carry it to the next rollout
                _a1_mem,
                a2_state,
                _a2_mem,
                env_state,
                env_params,
                agent_order,
            ) = vals
            traj_1, traj_2, env_states, a2_metrics = stack

            # reset second agent memory
            a2_mem = agent2.batch_reset(a2_mem, False)
            # jax.debug.breakpoint()
            traj_1_rewards = traj_1.rewards.mean(axis=(1,3))
            traj_2_rewards = traj_2.rewards.mean(axis=(1,3))
            for i in range(len(traj_1_rewards)):
                wandb.log({"r1": traj_1_rewards[i].item()}, step=i)
                wandb.log({"r2": traj_2_rewards[i].item()}, step=i)

            rewards_1 = jnp.concatenate([traj.rewards for traj in traj_1])
            rewards_2 = jnp.concatenate([traj.rewards for traj in traj_2])

            # logging
            self.train_episodes += 1
            if i % log_interval == 0:
                print(f"Episode {i}/{num_episodes}")
                if self.args.env_id == "coin_game":
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(),
                        self.cg_stats(env_state),
                    )
                    rewards_1 = rewards_1.sum(axis=1).mean()
                    rewards_2 = rewards_2.sum(axis=1).mean()

                elif self.args.env_id == "Fishery":
                    env_stats = fishery_stats(
                        traj_1 + traj_2, self.args.num_players
                    )
                    rewards_1 = rewards_1.sum(axis=1).mean()
                    rewards_2 = rewards_2.sum(axis=1).mean()

                elif self.args.env_id == "Rice-N":
                    env_stats = rice_eval_stats(
                        traj_1 + traj_2, env_states, env
                    )
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.tolist(),
                        env_stats,
                    )
                    env_stats = env_stats | rice_stats(
                        traj_1 + traj_2,
                        self.args.num_players,
                        self.args.has_mediator,
                    )
                    rewards_1 = rewards_1.sum(axis=1).mean()
                    rewards_2 = rewards_2.sum(axis=1).mean()

                elif self.args.env_id == "C-Rice-N":
                    env_stats = c_rice_eval_stats(
                        traj_1 + traj_2, env_states, env
                    )
                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.tolist(),
                        env_stats,
                    )
                    env_stats = env_stats | c_rice_stats(
                        traj_1 + traj_2,
                        self.args.num_players,
                        self.args.has_mediator,
                    )
                    rewards_1 = rewards_1.sum(axis=1).mean()
                    rewards_2 = rewards_2.sum(axis=1).mean()

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

                # Due to the permutation and roles the rewards are not in order
                agent_indices = jnp.array(
                    [0] * self.args.agent1_roles + [1] * self.args.agent2_roles
                )
                agent_indices = agent_indices[agent_order]

                # Omit rich per timestep statistics for cleaner logging
                printable_env_stats = {
                    k: v
                    for k, v in env_stats.items()
                    if not k.startswith("states")
                }
                print(f"Env Stats: {printable_env_stats}")
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
                            "agent_order": agent_order,
                            "agent_indices": agent_indices,
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
