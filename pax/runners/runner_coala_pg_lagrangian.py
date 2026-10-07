"""Runner for COALA-PG on a constrained-welfare Lagrangian.

Solves the constrained welfare problem with one individual-rationality
constraint per player -- the same two constraints as
`runner_welfare_evo` (there: ``mu1``/``mu2`` against
``v_ref_shaper``/``v_ref_opponent``):

    max_theta  W(theta)
    s.t.       R_s(theta) >= tau_s        (the shaper)
               R_o(theta) >= tau_o        (the co-player)

with Lagrangian

    L(theta, lam) = W + lam_s * (R_s - tau_s) + lam_o * (R_o - tau_o).

Because each constraint is itself an expected return, every term is an
expectation over the SAME trajectories, so the Lagrangian gradient is just the
COALA-PG gradient of a reshaped reward:

    r_mix = (r_s + r_o) + lam_s * r_s + lam_o * r_o
          = (1 + lam_s) * r_s + (1 + lam_o) * r_o.

Nothing in the learning-aware estimator changes. Relative to
`runner_coala_pg.CoalaPGRunner`, exactly three things differ:

  1. the trajectory carries a welfare stream plus one stream per constraint,
  2. the baseline comes from one critic head per return, combined as
     ``V_W + sum_k lam_k * V_k``,
  3. one lam controller per constraint runs after each meta-trajectory.

The rollout structure, the co-player, and `coala_advantages` are untouched,
which is what lets a selfish / welfare / constrained comparison be attributed
to the objective rather than to the optimizer.

WHY TWO MULTIPLIERS AND NOT ONE. They do different jobs. ``lam_s`` pulls the
shaper back from self-sacrifice; ``lam_o`` protects the co-player from being
exploited. With only ``lam_s``, the co-player's sole defence is the welfare
term, and welfare is indifferent between (-10, -30) and (-30, -10) -- so the
"cooperation" the shaper converges to can be an exploitative split that happens
to sum well. Both constraints together are what make the claim "cooperation is
established in the asymmetric setting" mean individual rationality for BOTH
players.

Reference for the dual update: Stooke, Achiam & Abbeel, "Responsive Safety in
Reinforcement Learning by PID Lagrangian Methods", ICML 2020. Setting a
constraint's ``kp = kd = 0`` recovers RCPO (Tessler et al., 2019) dual ascent
exactly, so the dual-ascent baseline needs no separate runner.

Two modelling choices that matter more than the controller gains:

  * THE CONSTRAINT WINDOW. A floor on a player's average return over the whole
    meta-episode can be satisfied early and violated late, after the co-player
    has converged -- the opposite of "shaping established cooperation".
    ``window`` restricts a constraint to the last K inner episodes, making tau
    a statement about where the co-player ENDS UP.
  * THE SCALE OF v_ref. Constraint returns are normalised to a PER-INNER-EPISODE
    scale, so tau can be read straight off the payoff matrix (e.g. the
    mutual-defection payoff, which makes each constraint exactly the
    individual-rationality condition and "self-sacrifice" the well-defined
    event R < v_ref). If the ES/Shaper runs define their floor on total
    meta-episode return instead, rescale one of them so the constraints mean
    the same thing on both optimizers.

CONFIG. Flat `welfare.*` keys, with the reference values named exactly as in
the constrained-welfare ES runner, so one override sweeps a reference value
and it means the same thing everywhere:

    ++welfare.v_ref_shaper=-15 ++welfare.v_ref_opponent=-20

Every other setting (`kp`, `ki`, `kd`, `lam_init`, `lam_min`, `lam_max`, `ema_beta`,
`constraint_window`) may be given once for both players or per player with a
`_shaper` / `_opponent` suffix, the suffixed form winning:

    ++welfare.ki=0.005                 # both
    ++welfare.ki_opponent=0.002        # just the co-player's dual

`++welfare.constrain_opponent=False` drops a constraint entirely, which is the
ablation showing both constraints are load-bearing.
"""

import os
import time
from datetime import datetime
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import wandb

from pax.runners.pid_lagrangian import (
    PIDLagrangian,
    parse_constraint_specs,
)
from pax.utils import MemoryState, TrainingState, save
from pax.watchers import cg_visitation, ipd_visitation

MAX_WANDB_CALLS = 1000

class Sample(NamedTuple):
    """A batch of data.

    `rewards` is the welfare stream. `cost_rewards` holds the K constraint
    streams on a trailing axis (each already masked to its window and
    rescaled). `behavior_values` carries ``1 + K`` critic heads, welfare first.
    """

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    cost_rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


def to_long_trajectory(traj: Sample) -> Sample:
    """``[M, T, num_opps, num_envs, ...]`` -> ``[M*T, num_opps, num_envs, ...]``.

    Keeps ``num_opps`` and ``num_envs`` separate: collapsing them would make
    the COALA cross-batch average run over *independent* co-players, which is
    wrong -- the coupling exists only within one co-player's minibatch.
    """
    return jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), traj
    )


def constraint_window_mask(
    num_outer_steps: int, window: int, dtype=jnp.float32
) -> jnp.ndarray:
    """Weights over the M inner episodes selecting one constraint's window.

    Returns ``[M]``, zero outside the last ``window`` episodes and
    ``M / window`` inside it. The scale is chosen so that

        sum_m mask[m] * ep_return[m] / M  ==  mean of ep_return over the window

    i.e. the constraint stream integrates to the windowed per-episode mean
    multiplied by M -- the same scale as the unwindowed return. With
    ``window >= M`` the mask is all ones and the stream is exactly the raw
    reward, so ``r_mix = (1 + lam_s) r_s + (1 + lam_o) r_o`` with no hidden
    factor.
    """
    if window <= 0 or window >= num_outer_steps:
        return jnp.ones((num_outer_steps,), dtype=dtype)
    idx = jnp.arange(num_outer_steps)
    inside = idx >= (num_outer_steps - window)
    return jnp.where(inside, num_outer_steps / float(window), 0.0).astype(dtype)


class CoalaPGLagrangianRunner:
    """Trains a Lagrangian-constrained COALA-PG shaper against a learner."""

    def __init__(self, agents, env, save_dir, args):
        self.args = args
        self.env = env
        self.save_dir = save_dir
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_time = time.time()
        self.start_datetime = datetime.now()
        self.train_steps = 0
        self.train_episodes = 0
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)

        num_outer_steps = args.num_outer_steps
        # Flat `welfare.*` keys, reference values named exactly as in the
        # constrained-welfare ES configs
        # (`v_ref_shaper`, `v_ref_opponent`) so a sweep override is one token.
        self.constraints = parse_constraint_specs(args.welfare)
        self.num_constraints = len(self.constraints)
        self.controllers = [
            PIDLagrangian(
                tau=c["tau"],
                kp=c["kp"],
                ki=c["ki"],
                kd=c["kd"],
                lam_init=c["lam_init"],
                lam_max=c["lam_max"],
                ema_beta=c["ema_beta"],
                lam_min=c["lam_min"],
            )
            for c in self.constraints
        ]
        # Fixed multipliers turn this into the static weighted-welfare
        # ablation: r_mix = (1+lam_s) r_s + (1+lam_o) r_o with lam never
        # updated. Worth running, because a reviewer will ask whether fixed
        # weights do the same job.
        self.freeze_lam = bool(args.welfare.get("freeze_lam", False))
        self.lam = jnp.asarray(
            [c.lam for c in self.controllers], dtype=jnp.float32
        )

        # [K, M] -- one window mask per constraint.
        window_masks = jnp.stack(
            [
                constraint_window_mask(num_outer_steps, c["window"])
                for c in self.constraints
            ],
            axis=0,
        )
        self._window_masks = window_masks
        # Which reward stream each constraint reads.
        player_index = tuple(c["reward_index"] for c in self.constraints)

        # ---- VMAP env over num_envs, then num_opps ----
        env.batch_reset = jax.vmap(env.reset, (0, None), 0)
        env.batch_step = jax.vmap(env.step, (0, 0, 0, None), 0)
        env.batch_reset = jax.jit(jax.vmap(env.batch_reset, (0, None), 0))
        env.batch_step = jax.jit(jax.vmap(env.batch_step, (0, 0, 0, None), 0))

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))

        agent1, agent2 = agents
        num_constraints = self.num_constraints

        # ---- agent 1 (Lagrangian COALA-PG shaper): batched over num_opps ----
        agent1.batch_init = jax.vmap(
            agent1.make_initial_state, (None, 0), (None, 0)
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # ---- agent 2 (co-player): batched over num_opps ----
        agent2.batch_init = jax.vmap(agent2.make_initial_state, (0, None), 0)
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )
        # in_axes=(1, 0, 0, 0): the trajectory's num_opps axis is 1, so each
        # co-player's update consumes all num_envs (= B) trajectories.
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
        agent1._state, agent1._mem = agent1.batch_init(
            agent1._state.random_key, init_hidden
        )

        a2_rng = jax.random.split(agent2._state.random_key, args.num_opps)
        agent2._state, agent2._mem = agent2.batch_init(
            a2_rng, jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
        )

        # Naive-learner population (paper, App. B.2.2 and Table 2,
        # `population_size (naive)` = 10, `dynamic_naive_agents` = False): the
        # co-player is always (re-)initialised from a FIXED set of
        # `naive_population_size` parameter vectors, sampled uniformly with
        # replacement for each of the num_opps co-players of a meta-trajectory.
        # Sampling a brand-new initialisation every time (the previous
        # behaviour, `naive_population_size: 0`) widens the task distribution
        # the shaper has to cover and is not what the paper trained against.
        naive_pop = int(args.get("naive_population_size", 10))
        self._naive_pool_keys = (
            jax.random.split(
                jax.random.PRNGKey(args.seed + 7919), naive_pop
            )
            if naive_pop > 0
            else None
        )
        pool_keys = self._naive_pool_keys

        def _inner_rollout(carry, unused):
            """One step of an inner episode."""
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
                env_params,
                env_state,
            ) = carry

            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            rngs = rngs[:, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state, obs1, a1_mem
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state, obs2, a2_mem
            )
            (next_obs1, next_obs2), env_state, rewards, done, info = (
                env.batch_step(env_rng, env_state, (a1, a2), env_params)
            )

            # The K constraint streams, before window masking: constraint k
            # reads player `player_index[k]`'s reward.
            cost = jnp.stack(
                [rewards[idx] for idx in player_index], axis=-1
            )
            traj1 = Sample(
                obs1,
                a1,
                rewards[0] + rewards[1],
                cost,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                done,
                a1_mem.hidden,
            )
            traj2 = Sample(
                obs2,
                a2,
                rewards[1],
                cost,
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
                env_params,
                env_state,
            ), (traj1, traj2, rewards[0], rewards[1])

        def _outer_rollout(carry, unused):
            """One inner episode, followed by the co-player's learning step."""
            vals, (traj1, traj2, raw_r1, raw_r2) = jax.lax.scan(
                _inner_rollout, carry, None, length=self.args.num_inner_steps
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
                env_params,
                env_state,
            ) = vals

            # Co-player learns from its minibatch of all num_envs trajectories.
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                traj2, obs2, a2_state, a2_mem
            )
            # a1_mem is intentionally NOT reset: carrying the GRU state across
            # the episode boundary is what gives the shaper its long-context
            # view of the co-player's learning dynamics.
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
                env_params,
                env_state,
            ), (traj1, traj2, a2_metrics, raw_r1, raw_r2)

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _a1_mem: MemoryState,
            _a2_state: TrainingState,
            _a2_mem: MemoryState,
            _env_params: Any,
            _lam: jnp.ndarray,
        ):
            """One meta-trajectory: M inner episodes against a fresh co-player."""
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.batch_reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
            ]

            _a1_mem = agent1.batch_reset(_a1_mem, False)
            if pool_keys is None:
                a2_rng = jax.random.split(_rng_run, args.num_opps)
            else:
                idx = jax.random.randint(
                    _rng_run, (args.num_opps,), 0, pool_keys.shape[0]
                )
                a2_rng = pool_keys[idx]
            _a2_state, _a2_mem = agent2.batch_init(a2_rng, _a2_mem.hidden)

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
                    _env_params,
                    env_state,
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
                _env_params,
                env_state,
            ) = vals
            traj_1, traj_2, a2_metrics, raw_r1, raw_r2 = stack

            # Per-episode returns: [M, T, num_opps, num_envs] -> [M]
            ep_rewards_1 = raw_r1.sum(axis=1).mean(axis=(1, 2))
            ep_rewards_2 = raw_r2.sum(axis=1).mean(axis=(1, 2))
            ep_rewards = jnp.stack([ep_rewards_1, ep_rewards_2], axis=0)

            # Apply each constraint's window to its own stream.
            # window_masks is [K, M]; cost_rewards is [M, T, opps, envs, K].
            mask = jnp.transpose(window_masks, (1, 0)).reshape(
                (num_outer_steps, 1, 1, 1, num_constraints)
            )
            traj_1 = traj_1._replace(cost_rewards=traj_1.cost_rewards * mask)

            # What each controller regulates: that player's mean
            # per-inner-episode return over that constraint's window, on the
            # same scale as its tau.
            constrained_returns = jnp.stack(
                [
                    jnp.sum(window_masks[k] * ep_rewards[player_index[k]])
                    / float(num_outer_steps)
                    for k in range(num_constraints)
                ],
                axis=0,
            )

            long_traj_1 = to_long_trajectory(traj_1)
            a1_state, a1_mem, a1_metrics = agent1.update(
                long_traj_1, obs1, a1_state, a1_mem, _lam
            )

            if args.env_id == "iterated_matrix_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(traj_1.observations, traj_1.actions, obs1),
                )
            else:
                env_stats = {}

            return (
                env_stats,
                ep_rewards_1,
                ep_rewards_2,
                constrained_returns,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            )

        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, num_iters, watchers):
        print("Training COALA-PG (Lagrangian constrained welfare)")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        M = self.args.num_outer_steps
        print(f"Number of meta-trajectories (iterations): {num_iters}")
        print(f"Inner episodes per meta-trajectory (M): {M}")
        print(f"Inner episode length (T): {self.args.num_inner_steps}")
        print(f"Co-player batch (B = num_envs): {self.args.num_envs}")
        print(f"Independent co-players (num_opps): {self.args.num_opps}")
        print(f"Log interval: {log_interval}")
        print(
            f"Constrained welfare: W + sum_k lam_k (R_k - tau_k), "
            f"{self.num_constraints} constraint(s)"
        )
        for c, ctrl in zip(self.constraints, self.controllers):
            window = (
                "whole meta-episode"
                if c["window"] <= 0 or c["window"] >= M
                else f"last {c['window']} of {M} episodes"
            )
            rcpo = (
                "  (kp=kd=0 => RCPO dual ascent)"
                if c["kp"] == 0 and c["kd"] == 0
                else ""
            )
            print(
                f"  [{c['name']}] v_ref_{c['suffix']}={c['tau']:.4f}  "
                f"window: {window}"
            )
            print(
                f"      kp={c['kp']} ki={c['ki']} kd={c['kd']} "
                f"lam_init={c['lam_init']} lam_min={c['lam_min']} "
                f"lam_max={c['lam_max']} "
                f"ema_beta={c['ema_beta']}{rcpo}"
            )
        if self.freeze_lam:
            print("  [FROZEN: static weighted-welfare ablation]")
        print("  tau is per inner episode -- read it off the payoff matrix.")

        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            lam_used = self.lam
            (
                env_stats,
                ep_rewards_1,
                ep_rewards_2,
                constrained_returns,
                a1_state,
                a1_mem,
                a1_metrics,
                a2_state,
                a2_mem,
                a2_metrics,
            ) = self.rollout(
                rng_run,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_params,
                lam_used,
            )

            # ---- dual updates, on the host, after the primal step ----
            r_constrained = [float(x) for x in constrained_returns]
            if self.freeze_lam:
                for ctrl, r in zip(self.controllers, r_constrained):
                    ctrl.delta = ctrl.tau - r
            else:
                self.lam = jnp.asarray(
                    [
                        ctrl.update(r)
                        for ctrl, r in zip(self.controllers, r_constrained)
                    ],
                    dtype=jnp.float32,
                )

            if i % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            if i % log_interval == 0:
                first_1, last_1 = float(ep_rewards_1[0]), float(ep_rewards_1[-1])
                first_2, last_2 = float(ep_rewards_2[0]), float(ep_rewards_2[-1])
                lam_used_f = [float(x) for x in lam_used]
                lam_next_f = [float(x) for x in self.lam]
                print(f"Iteration {i}")
                print(
                    f"  episode 1   : shaper {first_1:.4f} | co-player {first_2:.4f}"
                )
                print(
                    f"  episode {M:<3} : shaper {last_1:.4f} | co-player {last_2:.4f}"
                )
                print(
                    f"  meta-mean   : shaper {float(ep_rewards_1.mean()):.4f} | "
                    f"co-player {float(ep_rewards_2.mean()):.4f} | "
                    f"welfare {float(ep_rewards_1.mean() + ep_rewards_2.mean()):.4f}"
                )
                meta_means = [
                    float(ep_rewards_1.mean()),
                    float(ep_rewards_2.mean()),
                ]
                n_violated = 0
                n_ir_violated = 0
                for k, (c, ctrl) in enumerate(
                    zip(self.constraints, self.controllers)
                ):
                    viol = ctrl.tau - r_constrained[k]
                    n_violated += viol > 0
                    # R_k is the WINDOWED quantity the dual actually regulates.
                    print(
                        f"  [{c['name']:<9}] R {r_constrained[k]:8.4f} vs tau "
                        f"{ctrl.tau:8.4f} | viol {viol:+8.4f} | "
                        f"{'VIOLATED ' if viol > 0 else 'satisfied'} | "
                        f"lam {lam_used_f[k]:.4f} -> {lam_next_f[k]:.4f} "
                        f"(I {ctrl._integral:.4f} P {ctrl.p_term:+.4f} "
                        f"D {ctrl.d_term:+.4f})"
                    )
                    # Individual rationality is a statement about the WHOLE
                    # meta-episode: the alternative to shaping (always defect)
                    # pays v_ref in every episode, early ones included. So
                    # always report it, whatever window the dual uses -- a
                    # windowed constraint can read "satisfied" while the full
                    # meta-episode violates, and reporting only the former
                    # would claim IR that does not hold. Measured at
                    # window=5: tail -19.03 "satisfied" vs meta-mean -21.59.
                    ir_viol = ctrl.tau - meta_means[c["reward_index"]]
                    n_ir_violated += ir_viol > 0
                    if c["window"] > 0 and c["window"] < M:
                        flag = (
                            "  <-- WINDOW HIDES AN IR VIOLATION"
                            if ir_viol > 0 and viol <= 0
                            else ""
                        )
                        print(
                            f"      full-meta IR: {meta_means[c['reward_index']]:8.4f}"
                            f" vs {ctrl.tau:8.4f} | viol {ir_viol:+8.4f} | "
                            f"{'VIOLATED' if ir_viol > 0 else 'satisfied'}"
                            f"{flag}"
                        )
                # The effective reward weights the policy actually saw. Watch
                # for either weight running away: lam_k = 4 already means a 5x
                # weight on that player, i.e. the welfare term is gone.
                w_s = 1.0 + sum(
                    lam_used_f[k]
                    for k, c in enumerate(self.constraints)
                    if c["reward_index"] == 0
                )
                w_o = 1.0 + sum(
                    lam_used_f[k]
                    for k, c in enumerate(self.constraints)
                    if c["reward_index"] == 1
                )
                print(
                    f"  weights     : (1+lam_s) = {w_s:.4f} on r_s | "
                    f"(1+lam_o) = {w_o:.4f} on r_o | ratio {w_s / w_o:.3f}"
                    f" | feasible: {self.num_constraints - n_violated}"
                    f"/{self.num_constraints}"
                    f" | full-meta IR: {self.num_constraints - n_ir_violated}"
                    f"/{self.num_constraints}"
                )
                for stat, val in env_stats.items():
                    print(f"  {stat}: {float(val)}")
                print()

                if watchers:
                    flat_a1 = jax.tree_util.tree_map(jnp.mean, a1_metrics)
                    agent1._logger.metrics = agent1._logger.metrics | flat_a1
                    flat_a2 = jax.tree_util.tree_map(
                        lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                    )
                    agent2._logger.metrics = agent2._logger.metrics | flat_a2
                    for watcher, agent in zip(watchers, agents):
                        watcher(agent)

                    per_constraint = {}
                    for k, (c, ctrl) in enumerate(
                        zip(self.constraints, self.controllers)
                    ):
                        tag = c["name"].replace("-", "_")
                        viol = ctrl.tau - r_constrained[k]
                        per_constraint[
                            f"train/lagrangian/{tag}/constrained_return"
                        ] = r_constrained[k]
                        per_constraint[f"train/lagrangian/{tag}/lam_used"] = (
                            lam_used_f[k]
                        )
                        per_constraint[f"train/lagrangian/{tag}/violation"] = (
                            viol
                        )
                        per_constraint[f"train/lagrangian/{tag}/feasible"] = (
                            float(viol <= 0)
                        )
                        per_constraint[f"train/lagrangian/{tag}/tau"] = ctrl.tau
                        per_constraint[f"train/lagrangian/{tag}/integral"] = (
                            ctrl._integral
                        )
                        per_constraint[f"train/lagrangian/{tag}/p_term"] = (
                            ctrl.p_term
                        )
                        per_constraint[f"train/lagrangian/{tag}/d_term"] = (
                            ctrl.d_term
                        )
                        # Full-meta-episode IR, independent of the window the
                        # dual regulates. THIS is the quantity the paper's
                        # "no self-sacrifice / no exploitation" claim rests on.
                        ir_viol = (
                            ctrl.tau - meta_means[c["reward_index"]]
                        )
                        per_constraint[
                            f"train/lagrangian/{tag}/ir_full_meta_return"
                        ] = meta_means[c["reward_index"]]
                        per_constraint[
                            f"train/lagrangian/{tag}/ir_full_meta_violation"
                        ] = ir_viol
                        per_constraint[
                            f"train/lagrangian/{tag}/ir_full_meta_feasible"
                        ] = float(ir_viol <= 0)

                    wandb.log(
                        {
                            "train_iteration": i,
                            "train/reward_per_episode/player_1": float(
                                ep_rewards_1.mean()
                            ),
                            "train/reward_per_episode/player_2": float(
                                ep_rewards_2.mean()
                            ),
                            "train/welfare": float(
                                ep_rewards_1.mean() + ep_rewards_2.mean()
                            ),
                            "train/first_episode/player_1": first_1,
                            "train/first_episode/player_2": first_2,
                            "train/final_episode/player_1": last_1,
                            "train/final_episode/player_2": last_2,
                            "train/shaping_delta/player_2": last_2 - first_2,
                            # Effective reward weights, and whether every
                            # constraint is satisfied at once -- the thing the
                            # "cooperation is established" claim rests on.
                            "train/lagrangian/self_weight": w_s,
                            "train/lagrangian/opponent_weight": w_o,
                            "train/lagrangian/weight_ratio": w_s / w_o,
                            "train/lagrangian/all_feasible": float(
                                n_violated == 0
                            ),
                            "train/lagrangian/num_violated": float(n_violated),
                            # Both players individually rational over the
                            # whole meta-episode -- the headline condition.
                            "train/lagrangian/all_ir_full_meta": float(
                                n_ir_violated == 0
                            ),
                            "train/lagrangian/num_ir_violated": float(
                                n_ir_violated
                            ),
                        }
                        | per_constraint
                        | {k2: float(v) for k2, v in env_stats.items()}
                        | {
                            f"train/shaper/{k2}": float(v)
                            for k2, v in flat_a1.items()
                        },
                    )

        agents[0]._state = a1_state
        agents[1]._state = a2_state
        return agents
