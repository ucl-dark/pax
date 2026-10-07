"""Welfare-maximising Evolutionary Runner — unconstrained baseline.

Same rollout and ES logic as WelfareEvoRunner, but fitness is pure welfare
(R1 + R2) with no IR constraints, no Lagrangian multipliers, and no dual ascent.
Intended as a contrast run to isolate the effect of the constraints.
"""

import os
import pickle
import glob as glob_mod
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


class WelfareUnconstrainedEvoRunner:
    """ES runner that maximises joint welfare with no IR constraints.

    Identical to WelfareEvoRunner except fitness = welfare (R1 + R2) directly,
    with no Lagrangian multipliers (mu1, mu2) or penalty terms (rho1, rho2).

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
        # Vmap the environment
        # ------------------------------------------------------------------
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(env.step, (0, 0, 0, None), 0)
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(env.step, (0, 0, 0, None), 0)
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

        self._a1_init_hidden = agent1._mem.hidden

        # ------------------------------------------------------------------
        # Vmap agents
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

        strategy.ask = jax.jit(strategy.ask)
        strategy.tell = jax.jit(strategy.tell)

        # ------------------------------------------------------------------
        # Inner / outer rollout
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

            if args.agent1 in ["WelfareShaper", "MFOS"]:
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

            rewards_1_per_member = traj_1.rewards.sum(axis=1).mean(axis=(0, 2, 3))
            rewards_2_per_member = traj_2.rewards.sum(axis=1).mean(axis=(0, 2, 3))
            welfare_per_member = rewards_1_per_member + rewards_2_per_member

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
    # Checkpoint save / load
    # ------------------------------------------------------------------

    @staticmethod
    def _find_latest_resume(resume_dir):
        pattern = os.path.join(resume_dir, "**", "generation_*_resume")
        candidates = glob_mod.glob(pattern, recursive=True)
        if not candidates:
            return None

        best_gen = -1
        best_path = None
        for path in candidates:
            basename = os.path.basename(path)
            match = re.search(r"generation_(\d+)_resume$", basename)
            if match:
                gen_num = int(match.group(1))
                if gen_num > best_gen:
                    best_gen = gen_num
                    best_path = path
        return best_path

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
        """Run unconstrained welfare training (pure ES on welfare objective)."""

        resume_dir = getattr(self.args, 'resume_dir', "")
        resume_path = None
        if resume_dir and os.path.isdir(resume_dir):
            resume_path = self._find_latest_resume(resume_dir)
            if resume_path:
                print(f"[Resume] Found checkpoint: {resume_path}")
            else:
                print(f"[Resume] No generation_*_resume files in {resume_dir}, starting fresh.")

        print("Training (Welfare — unconstrained)")
        print("------------------------------")
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)
        print(f"Number of Generations: {num_iters}")
        print(f"Number of Meta Episodes: {self.num_outer_steps}")
        print(f"Population Size: {self.popsize}")
        print(f"Number of Environments: {self.args.num_envs}")
        print(f"Number of Opponents: {self.args.num_opps}")
        print(f"Log Interval: {log_interval}")
        print("------------------------------")

        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)

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

        start_gen = 0
        if resume_path:
            ckpt = load(resume_path)
            start_gen = ckpt["gen"] + 1
            rng = ckpt["rng"]
            evo_state = ckpt["evo_state"]
            log = ckpt["log"]
            print(f"[Resume] Resuming from generation {start_gen}/{num_iters}")

        for gen in range(start_gen, num_iters):
            rng, rng_run, rng_evo, rng_key = jax.random.split(rng, 4)

            x, evo_state = strategy.ask(rng_evo, evo_state, es_params)
            params = jax.vmap(param_reshaper.reshape_single)(x)
            params = jax.tree_util.tree_map(
                lambda p: p.reshape(
                    (self.args.num_devices, self.popsize) + p.shape[1:]
                ),
                params,
            )

            (
                welfare_per_member,
                r1_per_member,
                r2_per_member,
                env_stats,
                rewards_1,
                rewards_2,
                a2_metrics,
            ) = self.rollout(params, rng_run, a1_state, a1_mem, env_params)

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

            # ---- Unconstrained fitness: pure welfare ----
            fitness = welfare_per_member

            fitness_re = fit_shaper.apply(x, fitness)
            if self.args.es.mean_reduce:
                fitness_re = fitness_re - fitness_re.mean()
            evo_state = strategy.tell(x, fitness_re, evo_state, es_params)

            log = es_logging.update(log, x, fitness)

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

                resume_savepath = os.path.join(self.save_dir, f"generation_{gen}_resume")
                resume_data = {
                    "gen": gen,
                    "rng": rng,
                    "evo_state": evo_state,
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
                    "train/fitness/welfare": float(fitness.mean()),
                    "train/fitness/player_1": float(r1_per_member.mean()),
                    "train/fitness/player_2": float(r2_per_member.mean()),
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
