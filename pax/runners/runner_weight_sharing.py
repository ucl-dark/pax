import os
import time
from typing import Any, List, Tuple

import jax
import jax.numpy as jnp
import wandb

from pax.utils import MemoryState, TrainingState, save, Sample
from pax.watchers import fishery_stats
from pax.watchers.rice import rice_stats
from pax.watchers.c_rice import c_rice_stats

MAX_WANDB_CALLS = 1000000

"""
This runner implements weight sharing by letting the agent assume each role in the environment per turn.
"""


class WeightSharingRunner:
    """Holds the runner's state."""

    id = "weight_sharing"

    def __init__(self, agent, env, save_dir, args):
        self.train_steps = 0
        self.train_episodes = 0
        self.start_time = time.time()
        self.args = args
        self.random_key = jax.random.PRNGKey(args.seed)
        self.save_dir = save_dir

        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.random.split, (0, None))
        # set up agent
        if args.agent1 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent.batch_init = jax.jit(jax.vmap(agent.make_initial_state))
        else:
            # batch MemoryState not TrainingState
            agent.batch_init = jax.jit(agent.make_initial_state)

        agent.batch_reset = jax.jit(agent.reset_memory, static_argnums=1)

        agent.batch_policy = jax.jit(agent._policy)

        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent._mem.hidden, (1))
            agent._state, agent._mem = agent.batch_init(
                agent._state.random_key, init_hidden
            )

        def _inner_rollout(carry, unused) -> Tuple[Tuple, List[Sample]]:
            """Runner for inner episode"""
            (
                rngs,
                obs,
                a1_state,
                memories,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 2)
            env_rng = rngs[:, 0, :]
            rngs = rngs[:, 1, :]

            actions = []
            new_memories = []
            for i in range(len(obs)):
                action, a1_state, new_a1_mem = agent.batch_policy(
                    a1_state,
                    obs[i],
                    memories[i],
                )
                actions.append(action)
                new_memories.append(new_a1_mem)

            next_obs, env_state, rewards, done, info = env.step(
                env_rng,
                env_state,
                tuple(actions),
                env_params,
            )

            trajectories = [
                Sample(
                    observation,
                    action,
                    reward * jnp.logical_not(done),
                    new_memory.extras["log_probs"],
                    new_memory.extras["values"],
                    done,
                    memory.hidden,
                )
                for observation, action, reward, memory, new_memory in zip(
                    obs, actions, rewards, memories, new_memories
                )
            ]

            return (
                rngs,
                next_obs,
                a1_state,
                new_memories,
                env_state,
                env_params,
            ), trajectories

        def _rollout(
            _rng_run: jnp.ndarray,
            _a1_state: TrainingState,
            _memories: List[MemoryState],
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)]
            ).reshape((args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            _memories = [agent.batch_reset(mem, False) for mem in _memories]

            # run trials
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                (
                    rngs,
                    obs,
                    _a1_state,
                    _memories,
                    env_state,
                    _env_params,
                ),
                None,
                length=args.num_steps,
            )

            (
                rngs,
                obs,
                _a1_state,
                _memories,
                env_state,
                env_params,
            ) = vals

            for i in range(len(trajectories)):
                _a1_state, _, _a1_metrics = agent.update(
                    trajectories[i],
                    obs[i],
                    _a1_state,
                    _memories[i],
                )

            # reset memory
            _memories = [agent.batch_reset(_mem, False) for _mem in _memories]

            # Stats
            rewards = []
            num_episodes = jnp.sum(trajectories[0].dones)
            for traj in trajectories:
                rewards.append(
                    jnp.where(
                        num_episodes != 0,
                        jnp.sum(traj.rewards) / num_episodes,
                        0,
                    )
                )
            env_stats = {}
            if args.env_id == "Rice-N":
                env_stats = rice_stats(
                    trajectories, args.num_players, args.has_mediator
                )
            elif args.env_id == "C-Rice-N":
                env_stats = c_rice_stats(
                    trajectories, args.num_players, args.has_mediator
                )
            elif args.env_id == "Fishery":
                env_stats = fishery_stats(trajectories, args.num_players)

            return (
                env_stats,
                rewards,
                _a1_state,
                _memories,
                _a1_metrics,
            )

        self.rollout = jax.jit(_rollout)

    def run_loop(self, env, env_params, agent, num_iters, watcher):
        """Run training of agent in environment"""
        print("Training")
        print("-----------------------")
        agent = agent
        rng, _ = jax.random.split(self.random_key)

        a1_state, a1_mem = agent._state, agent._mem
        memories = tuple([a1_mem for _ in range(self.args.num_players)])
        num_iters = max(int(num_iters / (self.args.num_envs)), 1)
        log_interval = max(num_iters / MAX_WANDB_CALLS, 5)

        print(f"Log Interval {log_interval}")
        print(f"Running for total iterations: {num_iters}")
        # run actual loop
        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            # RL Rollout
            (
                env_stats,
                rewards,
                a1_state,
                memories,
                a1_metrics,
            ) = self.rollout(rng_run, a1_state, memories, env_params)

            is_last_iter = i == num_iters - 1
            if i % self.args.save_interval == 0 or is_last_iter:
                log_savepath = os.path.join(self.save_dir, f"iteration_{i}")
                save(a1_state.params, log_savepath)
                if watcher:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath)
                else:
                    print(f"Saving iteration {i} locally")

            # logging
            self.train_episodes += 1
            if num_iters % log_interval == 0 or is_last_iter:
                print(f"Episode {i}/{num_iters}")

                print(f"Env Stats: {env_stats}")
                print(f"Total Episode Reward: {float(sum(rewards))}")
                print()

                if watcher:
                    # metrics [outer_timesteps]
                    flattened_metrics_1 = jax.tree_util.tree_map(
                        lambda x: jnp.mean(x), a1_metrics
                    )
                    agent._logger.metrics = (
                        agent._logger.metrics | flattened_metrics_1
                    )

                    watcher(agent)
                    wandb.log(
                        {
                            "episodes": self.train_episodes,
                        }
                        | env_stats,
                    )

        agent._state = a1_state
        return agent
