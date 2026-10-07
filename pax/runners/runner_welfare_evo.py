"""Welfare-maximising Evolutionary Runner with Lagrangian IR constraints.

Objective (Nash welfare):
    max_theta  E[ log(R_1) + log(R_2) ]

IR constraints (enforced via dual ascent):
    E[ mean_e R_i^e ]  >= v_bar_i      (shaper IR)
    E[ mean_e R_{-i}^e ] >= v_bar_{-i}  (opponent IR)

The runner has two phases:
  1. **Calibration** — run the opponent's learning algorithm against a fixed
     non-shaping policy for the shaper to estimate v_bar_i and v_bar_{-i}.
  2. **Training** — standard ES loop, but fitness is the Lagrangian:
         L = welfare + mu1*(R_shaper - v_bar_i) + mu2*(R_opp - v_bar_{-i})
     with dual ascent on mu1, mu2 after each generation.
"""

import glob as glob_mod
import os
import pickle
import re
import time
from datetime import datetime
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from evosax import FitnessShaper

import wandb
from pax.utils import MemoryState, TrainingState, save, load
from pax.watchers import ESLog, cg_visitation, ipd_visitation, ipditm_stats

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


class WelfareEvoRunner:
    """ES runner that maximises joint welfare subject to IR constraints
    enforced through Lagrangian relaxation with dual ascent.

    Args:
        agents: (shaper, opponent) pair.
        env: Meta-environment.
        strategy: evosax ES strategy.
        es_params: ES hyper-parameters.
        param_reshaper: evosax ParameterReshaper.
        save_dir: Where to save checkpoints.
        args: Hydra experiment config.
    """

    def __init__(
        self, agents, env, strategy, es_params, param_reshaper, save_dir, args
    ):
        self.args = args
        self.algo = args.es.algo
        self.es_params = es_params
        self.generations = 0
        self.num_opps = args.num_opps
        self.param_reshaper = param_reshaper
        self.popsize = args.popsize
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_datetime = datetime.now()
        self.save_dir = save_dir
        self.start_time = time.time()
        self.strategy = strategy
        self.top_k = args.top_k
        self.train_steps = 0
        self.train_episodes = 0
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(jax.vmap(cg_visitation))
        self.ipditm_stats = jax.jit(
            jax.vmap(ipditm_stats, in_axes=(0, 2, 2, None))
        )

        # ------------------------------------------------------------------
        # Lagrangian dual variables  (IR constraints)
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

        self.rho1 = args.welfare.rho1                    # e.g. 1.0
        self.rho2 = args.welfare.rho2                    # e.g. 1.0
        self.rho_max = args.welfare.rho_max            # e.g. 100.0
        self.rho_multiplier = args.welfare.rho_schedule # e.g. 1.5
        self.rho_patience = args.welfare.rho_patience  # e.g. 50 generations
        self.violation_counter_1 = 0
        self.violation_counter_2 = 0

        

        # ------------------------------------------------------------------
        # Vmap the environment (same pattern as EvoRunner)
        # ------------------------------------------------------------------
        # num envs
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(env.step, (0, 0, 0, None), 0)
        # num opps
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(env.step, (0, 0, 0, None), 0)
        # pop size
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(env.step, (0, 0, 0, None), 0)
        )
        self.split = jax.vmap(
            jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None)),
            (0, None),
        )

        self.num_outer_steps = args.num_outer_steps
        agent1, agent2 = agents

        # Save the original (un-batched) hidden state before any batch_init.
        # calibrate() and run_loop() both call batch_init, which updates
        # agent1._mem.hidden to shape (popsize, num_opps, ...).  Tiling that
        # again would double the popsize dimension, so we always tile from here.
        self._a1_init_hidden = agent1._mem.hidden

        # ------------------------------------------------------------------
        # Vmap agents  (identical to EvoRunner)
        # ------------------------------------------------------------------
        agent1.batch_init = jax.vmap(
            jax.vmap(agent1.make_initial_state, (None, 0), (None, 0)),
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
            )
        )

        if args.agent2 == "NaiveEx":
            agent2.batch_init = jax.jit(
                jax.vmap(jax.vmap(agent2.make_initial_state))
            )
        else:
            agent2.batch_init = jax.jit(
                jax.vmap(
                    jax.vmap(agent2.make_initial_state, (0, None), 0),
                    (0, None), 0,
                )
            )

        agent2.batch_policy = jax.jit(jax.vmap(jax.vmap(agent2._policy, 0, 0)))
        agent2.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent2.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )
        agent2.batch_update = jax.jit(
            jax.vmap(jax.vmap(agent2.update, (1, 0, 0, 0)), (1, 0, 0, 0)),
        )

        if args.agent2 != "NaiveEx":
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
            a2_rng = jnp.concatenate(
                [jax.random.split(agent2._state.random_key, args.num_opps)]
                * args.popsize
            ).reshape(args.popsize, args.num_opps, -1)
            agent2._state, agent2._mem = agent2.batch_init(a2_rng, init_hidden)

        # JIT evo operators
        strategy.ask = jax.jit(strategy.ask)
        strategy.tell = jax.jit(strategy.tell)

        # ------------------------------------------------------------------
        # Inner / outer rollout  (same as EvoRunner)
        # ------------------------------------------------------------------

        def _inner_rollout(carry, unused):
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ) = carry

            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, :, 0, :]
            rngs = rngs[:, :, :, 3, :]

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state, obs1, a1_mem,
            )
            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state, obs2, a2_mem,
            )
            (next_obs1, next_obs2), env_state, rewards, done, info = env.step(
                env_rng, env_state, (a1, a2), env_params,
            )

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
                _inner_rollout, carry, None, length=args.num_inner_steps,
            )
            (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ) = vals

            # Meta-action for welfare shaper (same as MFOS)
            if args.agent1 in ["WelfareShaper", "MFOS"]:
                a1_mem = agent1.meta_policy(a1_mem)

            # Update opponent
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], obs2, a2_state, a2_mem,
            )
            return (
                rngs, obs1, obs2, r1, r2,
                a1_state, a1_mem, a2_state, a2_mem,
                env_state, env_params,
            ), (*trajectories, a2_metrics)

        def _rollout(
            _params, _rng_run, _a1_state, _a1_mem, _env_params,
        ):
            env_rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
                * args.num_opps * args.popsize
            ).reshape((args.popsize, args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(env_rngs, _env_params)
            rewards = [
                jnp.zeros((args.popsize, args.num_opps, args.num_envs)),
                jnp.zeros((args.popsize, args.num_opps, args.num_envs)),
            ]

            _a1_state = _a1_state._replace(params=_params)
            _a1_mem = agent1.batch_reset(_a1_mem, False)

            if args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(obs[1])
            else:
                a2_rng = jnp.concatenate(
                    [jax.random.split(_rng_run, args.num_opps)] * args.popsize
                ).reshape(args.popsize, args.num_opps, -1)
                a2_state, a2_mem = agent2.batch_init(
                    a2_rng, agent2._mem.hidden,
                )

            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    env_rngs, *obs, *rewards,
                    _a1_state, _a1_mem, a2_state, a2_mem,
                    env_state, _env_params,
                ),
                None, length=self.num_outer_steps,
            )

            (
                env_rngs, obs1, obs2, r1, r2,
                _a1_state, _a1_mem, a2_state, a2_mem,
                env_state, _env_params,
            ) = vals
            traj_1, traj_2, a2_metrics = stack

            # ---- Compute per-member reward statistics ----
            # traj.rewards shape: [outer, inner, pop, opps, envs]
            # Sum over inner (axis=1) to get per-episode rewards,
            # then mean over outer, opps, envs → per pop-member scalar
            rewards_1_per_member = traj_1.rewards.sum(axis=1).mean(axis=(0, 2, 3))
            rewards_2_per_member = traj_2.rewards.sum(axis=1).mean(axis=(0, 2, 3))
            # utilitarian welfare: W = R1 + R2
            welfare_per_member = ( rewards_1_per_member + rewards_2_per_member)

            # Nash welfare: W = log(R1 + shift) + log(R2 + shift)
            # _shift = 100.0  # make this a config param: args.welfare.reward_shift
            # _eps = 1e-6
            # welfare_per_member = (
            #     jnp.log(jnp.maximum(rewards_1_per_member + _shift, _eps))
            #     + jnp.log(jnp.maximum(rewards_2_per_member + _shift, _eps))
            # )

            # Env stats (same as EvoRunner)
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x, self.cg_stats(env_state),
                )
                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()
            elif args.env_id == "iterated_matrix_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(traj_1.observations, traj_1.actions, obs1),
                )
                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()
            elif args.env_id == "InTheMatrix":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipditm_stats(
                        env_state, traj_1, traj_2, args.num_envs,
                    ),
                )
                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()
            else:
                env_stats = {}
                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()

            return (
                welfare_per_member,
                rewards_1_per_member,
                rewards_2_per_member,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
            )

        self.rollout = jax.pmap(
            _rollout, in_axes=(0, None, None, None, None),
        )

        print(
            f"Time to Compile Jax Methods: {time.time() - self.start_time} Seconds"
        )

    # ------------------------------------------------------------------
    # Checkpoint save / load  (builds on existing save_interval)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_resume(resume_dir):
        """Scan *resume_dir* recursively for ``generation_*_resume`` files
        and return the path with the highest generation number, or None.

        Expected layout (created by the bash copy step)::

            resume_dir/
            ├── 2026-03-28_14.32.01/
            │   ├── generation_0
            │   ├── generation_0_resume
            │   ├── generation_100
            │   └── generation_100_resume
            └── 2026-03-29_10.00.00/
                ├── generation_200
                └── generation_200_resume
        """
        pattern = os.path.join(resume_dir, "**", "generation_*_resume")
        candidates = glob_mod.glob(pattern, recursive=True)
        if not candidates:
            return None

        # Parse generation number from each path
        best_gen = -1
        best_path = None
        for path in candidates:
            basename = os.path.basename(path)  # e.g. "generation_600_resume"
            match = re.search(r"generation_(\d+)_resume$", basename)
            if match:
                gen_num = int(match.group(1))
                if gen_num > best_gen:
                    best_gen = gen_num
                    best_path = path
        return best_path

    # ------------------------------------------------------------------
    # Calibration phase
    # ------------------------------------------------------------------

    def calibrate(self, env_params, agents, num_episodes: int):
        """Run the opponent's learning against a *fixed* shaper policy to
        estimate reference returns v_bar_i and v_bar_{-i}.

        We do this by running the standard rollout with the initial (random)
        shaper parameters for ``num_episodes`` generations and recording
        the mean per-episode returns.
        """
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)
        strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper
        popsize = self.popsize
        num_opps = self.num_opps

        evo_state = strategy.initialize(rng, es_params)

        init_hidden = jnp.tile(
            self._a1_init_hidden, (popsize, num_opps, 1, 1),
        )
        a1_rng = jax.random.split(rng, popsize)
        agent1._state, agent1._mem = agent1.batch_init(a1_rng, init_hidden)
        a1_state, a1_mem = agent1._state, agent1._mem

        all_r1 = []
        all_r2 = []

        print(f"Calibration: running {num_episodes} episodes with fixed shaper ...")
        for ep in range(num_episodes):
            rng, rng_run, rng_evo = jax.random.split(rng, 3)
            x, evo_state = strategy.ask(rng_evo, evo_state, es_params)
            params = jax.vmap(param_reshaper.reshape_single)(x)
            params = jax.tree_util.tree_map(
                lambda p: p.reshape(
                    (self.args.num_devices, self.popsize) + p.shape[1:]
                ),
                params,
            )

            (
                welfare, r1_per_member, r2_per_member,
                _env_stats, r1_mean, r2_mean, _a2_metrics,
            ) = self.rollout(params, rng_run, a1_state, a1_mem, env_params)

            all_r1.append(float(r1_mean.mean()))
            all_r2.append(float(r2_mean.mean()))

        self.v_ref_shaper = sum(all_r1) / len(all_r1)
        self.v_ref_opponent = sum(all_r2) / len(all_r2)
        print(
            f"Calibration done.  v_ref_shaper={self.v_ref_shaper:.4f}  "
            f"v_ref_opponent={self.v_ref_opponent:.4f}"
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def run_loop(
        self,
        env_params,
        agents,
        num_iters: int,
        watchers: Callable,
    ):
        """Run training with Lagrangian dual ascent on IR constraints."""

        # ---- Check for checkpoint to resume from ----
        resume_dir = getattr(self.args.welfare, 'resume_dir', "")
        resume_path = None
        if resume_dir and os.path.isdir(resume_dir):
            resume_path = self._find_latest_resume(resume_dir)
            if resume_path:
                print(f"[Resume] Found checkpoint: {resume_path}")
            else:
                print(f"[Resume] No generation_*_resume files in {resume_dir}, starting fresh.")

        # ---- Step 0: calibration (skip on resume) ----
        if resume_path:
            print("[Resume] Skipping calibration — will restore from checkpoint.")
        elif self.calibration:
            self.calibrate(env_params, agents, self.calibration_episodes)
        else:
            print(
                f"Skipping calibration. Using provided v_ref:"
                f"\n  v_ref_shaper={self.v_ref_shaper:.4f},"
                f" v_ref_opponent={self.v_ref_opponent:.4f}"
            )

        print("Training (Welfare + Lagrangian IR)")
        print("------------------------------")
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Number of Generations: {num_iters}")
        print(f"Number of Meta Episodes: {self.num_outer_steps}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {self.args.num_envs}")
        print(f"Number of Opponents: {self.args.num_opps}")
        print(f"v_ref_shaper: {self.v_ref_shaper:.4f}")
        print(f"v_ref_opponent: {self.v_ref_opponent:.4f}")
        print(f"Dual LR: {self.dual_lr}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")

        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

        # Re-initialise evolution and agent batches for training
        strategy = self.strategy
        es_params = self.es_params
        param_reshaper = self.param_reshaper
        popsize = self.popsize
        num_opps = self.num_opps
        evo_state = strategy.initialize(rng, es_params)
        fit_shaper = FitnessShaper(
            maximize=self.args.es.maximise,
            centered_rank=self.args.es.centered_rank,
            w_decay=self.args.es.w_decay,
            z_score=self.args.es.z_score,
        )
        es_logging = ESLog(
            param_reshaper.total_params,
            num_iters,
            top_k=self.top_k,
            maximize=True,
        )
        log = es_logging.initialize()

        init_hidden = jnp.tile(
            self._a1_init_hidden, (popsize, num_opps, 1, 1),
        )
        a1_rng = jax.random.split(rng, popsize)
        agent1._state, agent1._mem = agent1.batch_init(a1_rng, init_hidden)
        a1_state, a1_mem = agent1._state, agent1._mem

        # ---- Resume from checkpoint if available ----
        start_gen = 0
        if resume_path:
            ckpt = load(resume_path)
            start_gen = ckpt["gen"] + 1
            rng = ckpt["rng"]
            evo_state = ckpt["evo_state"]
            self.mu1 = ckpt["mu1"]
            self.mu2 = ckpt["mu2"]
            self.v_ref_shaper = ckpt["v_ref_shaper"]
            self.v_ref_opponent = ckpt["v_ref_opponent"]
            self.rho1 = ckpt["rho1"]
            self.rho2 = ckpt["rho2"]
            self.violation_counter_1 = ckpt["violation_counter_1"]
            self.violation_counter_2 = ckpt["violation_counter_2"]
            log = ckpt["log"]
            print(
                f"[Resume] Resuming from generation {start_gen}/{num_iters}"
                f"  mu1={self.mu1:.4f}  mu2={self.mu2:.4f}"
            )

        for gen in range(start_gen, num_iters):
            rng, rng_run, rng_evo, rng_key = jax.random.split(rng, 4)

            # Ask ES for candidate parameters
            x, evo_state = strategy.ask(rng_evo, evo_state, es_params)
            params = jax.vmap(param_reshaper.reshape_single)(x)
            params = jax.tree_util.tree_map(
                lambda p: p.reshape(
                    (self.args.num_devices, self.popsize) + p.shape[1:]
                ),
                params,
            )

            # Rollout
            (
                welfare_per_member,
                r1_per_member,
                r2_per_member,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
            ) = self.rollout(params, rng_run, a1_state, a1_mem, env_params)

            # Flatten over devices
            welfare_per_member = jnp.reshape(
                welfare_per_member, popsize * self.args.num_devices
            )
            r1_per_member = jnp.reshape(
                r1_per_member, popsize * self.args.num_devices
            )
            r2_per_member = jnp.reshape(
                r2_per_member, popsize * self.args.num_devices
            )
            env_stats = jax.tree_util.tree_map(lambda x: x.mean(), env_stats)

            # ---- Lagrangian fitness ----
            # L = welfare + mu1*(R1 - v_ref_1) + mu2*(R2 - v_ref_2)
            slack_1 = r1_per_member - self.v_ref_shaper
            slack_2 = r2_per_member - self.v_ref_opponent   
            fitness = (
                welfare_per_member
                + self.mu1 * slack_1
                + self.mu2 * slack_2
                - (self.rho1 / 2.0) * jnp.maximum(0.0, -slack_1)**2
                - (self.rho2 / 2.0) * jnp.maximum(0.0, -slack_2)**2
            )
#---------------------------------------Nash welfare---
            # _shift = 100.0
            # fitness = (
            #     welfare_per_member
            #     + self.mu1 * (slack_1 / _shift)
            #     + self.mu2 * (slack_2 / _shift)
            #     - (self.rho1 / 2.0) * (jnp.maximum(0.0, -slack_1) / _shift)**2
            #     - (self.rho2 / 2.0) * (jnp.maximum(0.0, -slack_2) / _shift)**2
            # )

            # ---- Dual ascent on multipliers ----
            mean_r1 = float(r1_per_member.mean()) # mean over population → scalar, it is still episode return averaged over all envs, opps, outer steps
            mean_r2 = float(r2_per_member.mean())
            # mu_k <- max(0, mu_k - alpha * (R_k_bar - v_ref_k))
            self.mu1 = max(0.0, self.mu1 - self.dual_lr * (mean_r1 - self.v_ref_shaper))
            self.mu2 = max(0.0, self.mu2 - self.dual_lr * (mean_r2 - self.v_ref_opponent))

            # After dual ascent update each generation
            shaper_violated = mean_r1 < self.v_ref_shaper
            opp_violated = mean_r2 < self.v_ref_opponent

            if shaper_violated:
                self.violation_counter_1 += 1
            else:
                self.violation_counter_1 = 0
            if opp_violated:
                self.violation_counter_2 += 1
            else:
                self.violation_counter_2 = 0

            if self.violation_counter_1 >= self.rho_patience:
                self.rho1 = min(self.rho1 * self.rho_multiplier, self.rho_max)
                self.violation_counter_1 = 0
                print(f"[Lagrangian] rho1 (shaper) increased to {self.rho1:.4f}")
            if self.violation_counter_2 >= self.rho_patience:
                self.rho2 = min(self.rho2 * self.rho_multiplier, self.rho_max)
                self.violation_counter_2 = 0
                print(f"[Lagrangian] rho2 (opponent) increased to {self.rho2:.4f}")

            # ---- ES tell ----
            fitness_re = fit_shaper.apply(x, fitness)
            if self.args.es.mean_reduce:
                fitness_re = fitness_re - fitness_re.mean()
            evo_state = strategy.tell(x, fitness_re, evo_state, es_params)

            # ---- Logging ----
            log = es_logging.update(log, x, fitness)

            # Saving
            if gen % self.args.save_interval == 0:
                log_savepath = os.path.join(self.save_dir, f"generation_{gen}")
                top_params = jax.vmap(param_reshaper.reshape_single)(
                    log["top_gen_params"][0:1]
                )
                top_params = jax.tree_util.tree_map(lambda p: p[0], top_params)
                save(top_params, log_savepath)
                if watchers:
                    print(f"Saving generation {gen} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {gen} locally")

                # Save resume data alongside the existing generation file
                resume_savepath = os.path.join(self.save_dir, f"generation_{gen}_resume")
                resume_data = {
                    "gen": gen,
                    "rng": rng,
                    "evo_state": evo_state,
                    "mu1": self.mu1,
                    "mu2": self.mu2,
                    "v_ref_shaper": self.v_ref_shaper,
                    "v_ref_opponent": self.v_ref_opponent,
                    "rho1": self.rho1,
                    "rho2": self.rho2,
                    "violation_counter_1": self.violation_counter_1,
                    "violation_counter_2": self.violation_counter_2,
                    "log": log,
                    "wandb_run_id": wandb.run.id if wandb.run else None,
                }
                with open(resume_savepath, "wb") as f:
                    pickle.dump(resume_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            if gen % log_interval == 0:
                print(f"Generation: {gen}")
                print(
                    "--------------------------------------------------------------------------"
                )
                print(
                    f"Welfare: {float(welfare_per_member.mean()):.4f} | "
                    f"R1: {float(rewards_1.mean()):.4f} | R2: {float(rewards_2.mean()):.4f}"
                )
                print(
                    f"mu1 (shaper IR): {self.mu1:.4f} | mu2 (opp IR): {self.mu2:.4f}"
                )
                print(
                    f"v_ref_shaper: {self.v_ref_shaper:.4f} | "
                    f"v_ref_opponent: {self.v_ref_opponent:.4f}"
                )
                print(
                    f"Lagrangian fitness: {float(fitness.mean()):.4f}"
                )
                print(
                    f"Env Stats: {jax.tree_util.tree_map(lambda x: x.item(), env_stats)}"
                )
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
                for k in range(min(5, self.top_k)):
                    print(f"Agent {k+1} | Fitness: {log['top_gen_fitness'][k]}")
                print()

            if watchers:
                wandb_log = {
                    "train/reward_per_episode/player_1": float(rewards_1.mean()),
                    "train/reward_per_episode/player_2": float(rewards_2.mean()),
                    "train/reward_per_timestep/player_1": float(rewards_1.mean()) / self.args.num_inner_steps,
                    "train/reward_per_timestep/player_2": float(rewards_2.mean()) / self.args.num_inner_steps,
                    "train_iteration": gen,
                    "train/welfare/mean": float(welfare_per_member.mean()),
                    "train/fitness/lagrangian": float(fitness.mean()),
                    "train/fitness/player_1": float(r1_per_member.mean()),
                    "train/fitness/player_2": float(r2_per_member.mean()),
                    "train/lagrangian/mu1_shaper": self.mu1,
                    "train/lagrangian/mu2_opponent": self.mu2,
                    "train/lagrangian/v_ref_shaper": self.v_ref_shaper,
                    "train/lagrangian/v_ref_opponent": self.v_ref_opponent,
                    "train/lagrangian/constraint_slack_shaper": mean_r1 - self.v_ref_shaper,
                    "train/lagrangian/constraint_slack_opponent": mean_r2 - self.v_ref_opponent,
                    "train/fitness/top_overall_mean": log["log_top_mean"][gen],
                    "train/fitness/top_overall_std": log["log_top_std"][gen],
                    "train/fitness/top_gen_mean": log["log_top_gen_mean"][gen],
                    "train/fitness/top_gen_std": log["log_top_gen_std"][gen],
                    "train/fitness/gen_std": log["log_gen_std"][gen],
                    "train/time/minutes": float(
                        (time.time() - self.start_time) / 60
                    ),
                    "train/time/seconds": float(
                        (time.time() - self.start_time)
                    ),
                    "train/lagrangian/rho1_shaper": self.rho1,
                    "train/lagrangian/rho2_opponent": self.rho2,
                    "train/lagrangian/violation_counter_1": self.violation_counter_1,
                    "train/lagrangian/violation_counter_2": self.violation_counter_2,
                }
                wandb_log.update(env_stats)
                for idx, (overall_fitness, gen_fitness) in enumerate(
                    zip(log["top_fitness"], log["top_gen_fitness"])
                ):
                    wandb_log[
                        f"train/fitness/top_overall_agent_{idx+1}"
                    ] = overall_fitness
                    wandb_log[
                        f"train/fitness/top_gen_agent_{idx+1}"
                    ] = gen_fitness

                flattened_metrics = jax.tree_util.tree_map(
                    lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                )
                agent2._logger.metrics.update(flattened_metrics)
                for watcher, agent in zip(watchers, agents):
                    watcher(agent)
                wandb_log = jax.tree_util.tree_map(
                    lambda x: x.item() if isinstance(x, jax.Array) else x,
                    wandb_log,
                )
                wandb.log(wandb_log, step=gen)

        return agents
