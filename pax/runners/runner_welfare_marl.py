"""Welfare-maximising RL Runner with Lagrangian IR constraints.

Same objective as WelfareEvoRunner but agent1 (the shaper) is trained
via RL (PPO) rather than evolutionary strategies.

Objective:
    max_theta  E[ sum_e W^e ]   where  W^e = sum_t (r_i^{e,t} + r_{-i}^{e,t})

The shaper receives *augmented rewards*:
    r_aug_t = (r1_t + r2_t) + mu1 * r1_t + mu2 * r2_t

where mu1, mu2 are Lagrangian multipliers updated by dual ascent after
each trial to enforce individual-rationality constraints.
"""

import os
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.utils import MemoryState, TrainingState, save, copy_state_and_mem
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
    """Object containing a batch of data (with meta actions)"""
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray
    meta_actions: jnp.ndarray


@jax.jit
def reduce_outer_traj(traj):
    """Collapse lax.scan output dims."""
    num_envs = traj.rewards.shape[2] * traj.rewards.shape[3]
    num_timesteps = traj.rewards.shape[0] * traj.rewards.shape[1]
    return jax.tree_util.tree_map(
        lambda x: x.reshape((num_timesteps, num_envs) + x.shape[4:]),
        traj,
    )


class WelfareRLRunner:
    """RL runner that maximises joint welfare subject to IR constraints
    via Lagrangian relaxation with dual ascent.

    The shaper's reward signal is augmented in-place so that standard PPO
    training on that signal optimises the Lagrangian.
    """

    def __init__(self, agents, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.num_opps = args.num_opps
        self.random_key = jax.random.PRNGKey(args.seed)
        self.save_dir = save_dir

        # ------------------------------------------------------------------
        # Lagrangian dual variables
        # ------------------------------------------------------------------
        self.mu1 = args.welfare.mu1
        self.mu2 = args.welfare.mu2
        self.dual_lr = args.welfare.dual_lr
        self.calibration = args.welfare.calibration
        if self.calibration:
            self.calibration_episodes = args.welfare.calibration_episodes
            self.v_ref_shaper = 0.0
            self.v_ref_opponent = 0.0
        else:
            self.v_ref_shaper = args.welfare.v_ref_shaper
            self.v_ref_opponent = args.welfare.v_ref_opponent

        def _reshape_opp_dim(x):
            batch_size = args.num_envs * args.num_opps
            return jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), x
            )

        self.reduce_opp_dim = jax.jit(_reshape_opp_dim)
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)
        self.cournot_stats = cournot_stats
        self.ipditm_stats = jax.jit(ipditm_stats)

        # VMAP env
        env.batch_reset = jax.vmap(env.reset, (0, None), 0)
        env.batch_step = jax.vmap(env.step, (0, 0, 0, None), 0)
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, (0, None), 0))
        env.batch_step = jax.jit(
            jax.vmap(env.batch_step, (0, 0, 0, None), 0)
        )
        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))
        num_outer_steps = self.args.num_outer_steps
        agent1, agent2 = agents

        # Set up agent1
        if args.agent1 == "NaiveEx":
            agent1.batch_init = jax.jit(jax.vmap(agent1.make_initial_state))
        else:
            agent1.batch_init = jax.vmap(
                agent1.make_initial_state, (None, 0), (None, 0),
            )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # Set up agent2
        if args.agent2 == "NaiveEx":
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
            init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )
        if args.agent2 != "NaiveEx":
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
            a2_rng = jax.random.split(agent2._state.random_key, args.num_opps)
            agent2._state, agent2._mem = agent2.batch_init(a2_rng, init_hidden)

        def _inner_rollout(carry, unused):
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ) = carry

            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            rngs = rngs[:, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state, obs1, a1_mem,
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state, obs2, a2_mem,
            )
            (
                (next_obs1, next_obs2), env_state, rewards, done, info,
            ) = env.batch_step(env_rng, env_state, (a1, a2), env_params)

            if args.agent1 == "WelfareShaper":
                # MFOS-based: includes meta_actions (th)
                traj1 = MFOSSample(
                    obs1, a1, rewards[0],
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    done, a1_mem.hidden, a1_mem.th,
                )
            else:
                # Attention-based shaper or other: standard sample
                traj1 = Sample(
                    obs1, a1, rewards[0],
                    new_a1_mem.extras["log_probs"],
                    new_a1_mem.extras["values"],
                    done, a1_mem.hidden,
                )
            traj2 = Sample(
                obs2, a2, rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done, a2_mem.hidden,
            )
            return (
                rngs, next_obs1, next_obs2, rewards[0], rewards[1],
                a1_state, new_a1_mem, a2_state, new_a2_mem,
                env_state, env_params,
            ), (traj1, traj2)

        def _outer_rollout(carry, unused):
            vals, trajectories = jax.lax.scan(
                _inner_rollout, carry, None, length=self.args.num_inner_steps,
            )
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ) = vals
            # MFOS-based welfare shaper needs meta_policy between episodes
            # WelfareShaperAtt (Shaper) does not — shaping is via attention on hidden states
            if args.agent1 == "WelfareShaper":
                a1_mem = agent1.meta_policy(a1_mem)
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], obs2, a2_state, a2_mem,
            )
            return (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ), (*trajectories, a2_metrics)

        def _rollout(
            _rng_run, _a1_state, _a1_mem, _a2_state, _a2_mem, _env_params,
            _mu1, _mu2,
        ):
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.batch_reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
            ]
            _a1_mem = agent1.batch_reset(_a1_mem, False)

            if args.agent1 == "NaiveEx":
                _a1_state, _a1_mem = agent1.batch_init(obs[0])
            if args.agent2 == "NaiveEx":
                _a2_state, _a2_mem = agent2.batch_init(obs[1])
            elif self.args.env_type in ["meta"]:
                a2_rng = jax.random.split(_rng_run, self.num_opps)
                _a2_state, _a2_mem = agent2.batch_init(a2_rng, _a2_mem.hidden)

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs, *obs, *rewards,
                    _a1_state, _a1_mem, _a2_state, _a2_mem,
                    env_state, _env_params,
                ),
                None, length=num_outer_steps,
            )
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # ---- Augment shaper rewards with Lagrangian terms ----
            # r_aug = (r1 + r2) + mu1 * r1 + mu2 * r2
            # = (1 + mu1) * r1 + (1 + mu2) * r2
            # but value baseline was trained on r1 only, so we replace
            # the reward field entirely with the augmented signal
            augmented_rewards = (
                (1.0 + _mu1) * traj_1.rewards
                + (1.0 + _mu2) * traj_2.rewards
            )
            traj_1 = traj_1._replace(rewards=augmented_rewards)

            # Update shaper
            a1_state, _, a1_metrics = agent1.update(
                reduce_outer_traj(traj_1),
                self.reduce_opp_dim(obs1),
                a1_state,
                self.reduce_opp_dim(a1_mem),
            )

            a1_mem = agent1.batch_reset(a1_mem, False)
            a2_mem = agent2.batch_reset(a2_mem, False)

            rewards_1 = traj_1.rewards.mean()
            rewards_2 = traj_2.rewards.mean()
            # Original (non-augmented) rewards for constraint checking
            original_r1 = ((1.0 + _mu1) * traj_1.rewards).mean()  # already augmented
            # We need the raw r1. Recover from augmented:
            # Actually let's just track traj_2.rewards for r2,
            # and for r1 we can compute from welfare - r2
            raw_r1_mean = (augmented_rewards - (1.0 + _mu2) * traj_2.rewards).mean() / (1.0 + _mu1)
            raw_r2_mean = traj_2.rewards.mean()

            # Stats
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x, self.cg_stats(env_state),
                )
            elif args.env_id == "iterated_matrix_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(traj_1.observations, traj_1.actions, obs1),
                )
            elif args.env_id == "InTheMatrix":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipditm_stats(
                        env_state, traj_1, traj_2, args.num_envs,
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
                raw_r1_mean,
                raw_r2_mean,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            )

        self.rollout = jax.jit(_rollout)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, env_params, agents, num_episodes):
        """Estimate reference returns by running opponent against fixed shaper."""
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)
        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        all_r1 = []
        all_r2 = []

        print(f"Calibration: running {num_episodes} episodes with fixed shaper ...")
        for ep in range(num_episodes):
            rng, rng_run = jax.random.split(rng, 2)
            (
                _env_stats, _r1, _r2, raw_r1, raw_r2,
                a1_state, a1_mem, _a1_metrics,
                a2_state, a2_mem, _a2_metrics,
            ) = self.rollout(
                rng_run, a1_state, a1_mem, a2_state, a2_mem, env_params,
                0.0, 0.0,  # mu1=0, mu2=0 during calibration
            )
            all_r1.append(float(raw_r1))
            all_r2.append(float(raw_r2))

        self.v_ref_shaper = sum(all_r1) / len(all_r1)
        self.v_ref_opponent = sum(all_r2) / len(all_r2)
        print(
            f"Calibration done.  v_ref_shaper={self.v_ref_shaper:.4f}  "
            f"v_ref_opponent={self.v_ref_opponent:.4f}"
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def run_loop(self, env_params, agents, num_iters, watchers):
        # Calibrate first (or skip if v_ref provided in config)
        if self.calibration:
            self.calibrate(env_params, agents, self.calibration_episodes)
        else:
            print(
                f"Skipping calibration. Using provided v_ref:"
                f"\n  v_ref_shaper={self.v_ref_shaper:.4f},"
                f" v_ref_opponent={self.v_ref_opponent:.4f}"
            )

        print("Training (Welfare RL + Lagrangian IR)")
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
        print(f"v_ref_shaper: {self.v_ref_shaper:.4f}")
        print(f"v_ref_opponent: {self.v_ref_opponent:.4f}")
        print(f"Dual LR: {self.dual_lr}")

        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)

            (
                env_stats,
                rewards_1,
                rewards_2,
                raw_r1_mean,
                raw_r2_mean,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            ) = self.rollout(
                rng_run, a1_state, a1_mem, a2_state, a2_mem, env_params,
                self.mu1, self.mu2,
            )

            # Dual ascent
            r1_val = float(raw_r1_mean)
            r2_val = float(raw_r2_mean)
            self.mu1 = max(0.0, self.mu1 - self.dual_lr * (r1_val - self.v_ref_shaper))
            self.mu2 = max(0.0, self.mu2 - self.dual_lr * (r2_val - self.v_ref_opponent))

            # Saving
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

            # Logging
            if i % log_interval == 0:
                print(f"Episode {i}")
                print(
                    f"R1: {r1_val:.4f} | R2: {r2_val:.4f} | "
                    f"mu1: {self.mu1:.4f} | mu2: {self.mu2:.4f}"
                )
                for stat in env_stats.keys():
                    print(stat + f": {env_stats[stat].item()}")
                print(
                    f"Average Reward per Timestep: "
                    f"{float(rewards_1.mean()), float(rewards_2.mean())}"
                )
                print()

                if watchers:
                    flattened_metrics_1 = jax.tree_util.tree_map(
                        lambda x: jnp.mean(x), a1_metrics
                    )
                    agent1._logger.metrics = (
                        agent1._logger.metrics | flattened_metrics_1
                    )
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
                            "train/welfare/mean": float(rewards_1.mean() + rewards_2.mean()),
                            "train/reward_per_timestep/player_1": float(
                                rewards_1.mean().item()
                            ),
                            "train/reward_per_timestep/player_2": float(
                                rewards_2.mean().item()
                            ),
                            "train/lagrangian/mu1_shaper": self.mu1,
                            "train/lagrangian/mu2_opponent": self.mu2,
                            "train/lagrangian/v_ref_shaper": self.v_ref_shaper,
                            "train/lagrangian/v_ref_opponent": self.v_ref_opponent,
                            "train/lagrangian/constraint_slack_shaper": r1_val - self.v_ref_shaper,
                            "train/lagrangian/constraint_slack_opponent": r2_val - self.v_ref_opponent,
                        }
                        | env_stats,
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
