import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.utils import MemoryState, TrainingState, save
from pax.utils import (
    copy_state_and_mem,
)
from pax.watchers import cg_visitation, ipd_visitation, ipditm_stats
from pax.watchers.cournot import cournot_stats
from pax.watchers.fishery import fishery_stats

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


class MFOSSample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray
    meta_actions: jnp.ndarray


class LOLASample(NamedTuple):
    obs_self: jnp.ndarray
    obs_other: jnp.ndarray
    actions_self: jnp.ndarray
    actions_other: jnp.ndarray
    dones: jnp.ndarray
    rewards_self: jnp.ndarray
    rewards_other: jnp.ndarray


@jax.jit
def reduce_outer_traj(traj: Sample) -> Sample:
    """Used to collapse lax.scan outputs dims"""
    # x: [outer_loop, inner_loop, num_opps, num_envs ...]
    # x: [timestep, batch_size, ...]
    num_envs = traj.rewards.shape[2] * traj.rewards.shape[3]
    num_timesteps = traj.rewards.shape[0] * traj.rewards.shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_timesteps, num_envs) + x.shape[4:]),
        traj,
    )


class RLRunner:
    """
    Reinforcement Learning runner provides a convenient example for quickly writing
    a MARL runner for PAX. The MARLRunner class can be used to
    run any two RL agents together either in a meta-game or regular game, it composes together agents,
    watchers, and the environment. Within the init, we declare vmaps and pmaps for training.
    Args:
        agents (Tuple[agents]):
            The set of agents that will run in the experiment. Note, ordering is
            important for logic used in the class.
        env (gymnax.envs.Environment):
            The environment that the agents will run in.
        save_dir (string):
            The directory to save the model to.
        args (NamedTuple):
            A tuple of experiment arguments used (usually provided by HydraConfig).
    """

    # flake8: noqa: C901
    def __init__(self, agents, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opps
        self.random_key = jax.random.PRNGKey(args.seed)
        self.save_dir = save_dir

        def _reshape_opp_dim(x):
            # x: [num_opps, num_envs ...]
            # x: [batch_size, ...]
            batch_size = args.num_envs * args.num_opps
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), x
            )

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)
        self.cournot_stats = cournot_stats
        # VMAP for num_envs
        self.ipditm_stats = jax.jit(ipditm_stats)
        # VMAP for num envs: we vmap over the rng but not params
        env.batch_reset = jax.vmap(env.reset, (0, None), 0)
        env.batch_step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )

        # VMAP for num opps: we vmap over the rng but not params
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, (0, None), 0))
        env.batch_step = jax.jit(
            jax.vmap(
                env.batch_step,
                (0, 0, 0, None),
                0,  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))
        num_outer_steps = self.args.num_outer_steps
        agent1, agent2 = agents

        # set up agents
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
            a2_rng = jax.random.split(agent2._state.random_key, args.num_opps)
            agent2._state, agent2._mem = agent2.batch_init(
                a2_rng,
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
            (
                (next_obs1, next_obs2),
                env_state,
                rewards,
                done,
                info,
            ) = env.batch_step(
                env_rng,
                env_state,
                (a1, a2),
                env_params,
            )

            if args.agent1 == "MFOS":
                traj1 = MFOSSample(
                    obs1,
                    a1,
                    rewards[0],
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    done,
                    a1_mem.hidden,
                    a1_mem.th,
                )
            else:
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

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _a2_mem: MemoryState,
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.batch_reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
            ]
            # Player 1
            _a1_mem = agent1.batch_reset(_a1_mem, False)

            # Resetting Agents if necessary
            if args.agent1 == "NaiveEx":
                _a1_state, _a1_mem = agent1.batch_init(obs[0])

            if args.agent2 == "NaiveEx":
                _a2_state, _a2_mem = agent2.batch_init(obs[1])

            elif self.args.env_type in ["meta"]:
                # meta-experiments - init 2nd agent per trial
                a2_rng = jax.random.split(_rng_run, self.num_opps)
                _a2_state, _a2_mem = agent2.batch_init(a2_rng, _a2_mem.hidden)
            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    *obs,
                    *rewards,
                    _a1_state,
                    _a1_mem,
                    _a2_state,
                    _a2_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=num_outer_steps,
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

            # update outer agent
            if args.agent1 != "LOLA":
                a1_state, _, a1_metrics = agent1.update(
                    reduce_outer_traj(traj_1),
                    self.reduce_opp_dim(obs1),
                    a1_state,
                    self.reduce_opp_dim(a1_mem),
                )
            if args.agent1 == "LOLA":
                a1_metrics = None
                # copy so we don't modify the original during simulation
                self_state, self_mem = copy_state_and_mem(a1_state, a1_mem)
                other_state, other_mem = copy_state_and_mem(a2_state, a2_mem)
                # get new state of opponent after their lookahead optimisation
                for _ in range(args.lola.num_lookaheads):
                    _rng_run, _ = jax.random.split(_rng_run)
                    lookahead_rng = jax.random.split(_rng_run, args.num_opps)

                    # we want to batch this num_opps times
                    other_state, other_mem = agent1.batch_in_lookahead(
                        lookahead_rng,
                        self_state,
                        self_mem,
                        other_state,
                        other_mem,
                    )
                # get our new state after our optimisation based on ops new state
                _rng_run, out_look_rng = jax.random.split(_rng_run)
                a1_state = agent1.out_lookahead(
                    out_look_rng, a1_state, a1_mem, other_state, other_mem
                )

            if args.agent2 == "LOLA":
                raise NotImplementedError("LOLA not implemented for agent2")

            # reset memory
            a1_mem = agent1.batch_reset(a1_mem, False)
            a2_mem = agent2.batch_reset(a2_mem, False)

            rewards_1 = traj_1.rewards.mean()
            rewards_2 = traj_2.rewards.mean()
            # Stats
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x,
                    self.cg_stats(env_state),
                )

                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()
            elif args.env_id == "iterated_matrix_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        traj_1.observations,
                        traj_1.actions,
                        obs1,
                    ),
                )
            elif args.env_id == "InTheMatrix":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipditm_stats(
                        env_state,
                        traj_1,
                        traj_2,
                        args.num_envs,
                    ),
                )
            elif args.env_id == "Cournot":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.cournot_stats(traj_1.observations, _env_params, 2),
                )
            elif args.env_id == "Fishery":
                env_stats = fishery_stats([traj_1, traj_2], 2)
            else:
                env_stats = {}

            return (
                env_stats,
                rewards_1,
                rewards_2,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            )

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, num_iters, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        save_interval = self.args.save_interval

        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        print(f"Num iters {num_iters}")
        print(f"Log Interval {log_interval}")
        print(f"Save Interval {save_interval}")
        # run actual loop
        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            # RL Rollout
            (
                env_stats,
                rewards_1,
                rewards_2,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            ) = self.rollout(
                rng_run, a1_state, a1_mem, a2_state, a2_mem, env_params
            )

            # saving
            if i % save_interval == 0:
                log_savepath1 = os.path.join(
                    self.save_dir, f"agent1_iteration_{i}"
                )
                save(a1_state.params, log_savepath1)
                log_savepath2 = os.path.join(
                    self.save_dir, f"agent2_iteration_{i}"
                )
                save(a2_state.params, log_savepath2)
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath1)
                    wandb.save(log_savepath2)
                else:
                    print(f"Saving iteration {i} locally")

            # logging
            if i % log_interval == 0:
                print(f"Episode {i}")
                for stat in env_stats.keys():
                    print(stat + f": {env_stats[stat].item()}")
                print(
                    f"Average Reward per Timestep: {float(rewards_1.mean()), float(rewards_2.mean())}"
                )
                print()

                if watchers:
                    # metrics [outer_timesteps]
                    flattened_metrics_1 = jax.tree_util.tree_map(
                        lambda x: jnp.mean(x), a1_metrics
                    )
                    if self.args.agent1 != "LOLA":
                        agent1._logger.metrics = (
                            agent1._logger.metrics | flattened_metrics_1
                        )
                    # metrics [outer_timesteps, num_opps]
                    flattened_metrics_2 = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = (
                        agent2._logger.metrics | flattened_metrics_2
                    )

                    for watcher, agent in zip(watchers, agents):
                        watcher(agent)

                    env_stats = jax.tree_util.tree_map(
                        lambda x: x.item(), env_stats
                    )
                    wandb.log(
                        {
                            "train_iteration": i,
                            "train/reward_per_timestep/player_1": float(
                                rewards_1.mean().item()
                            ),
                            "train/reward_per_timestep/player_2": float(
                                rewards_2.mean().item()
                            ),
                        }
                        | env_stats,
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
