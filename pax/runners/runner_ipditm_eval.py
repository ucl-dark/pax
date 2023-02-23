import os
import time
from datetime import datetime
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as onp
from PIL import Image
from tqdm import tqdm

import wandb
from pax.envs.in_the_matrix import EnvState, InTheMatrix
from pax.utils import MemoryState, TrainingState, load
from pax.watchers import cg_visitation, ipd_visitation, ipditm_stats

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
    env_state: jnp.ndarray


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
    env_state: jnp.ndarray


class IPDITMEvalRunner:
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
        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, None), 0  # rng, state, actions, params
        )
        self.ipditm_stats = jax.jit(ipditm_stats)
        # VMAP for num opps: we vmap over the rng but not params
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step, (0, 0, 0, None), 0  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))
        self.num_outer_steps = self.args.num_outer_steps

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
                env_state,
            )
            traj2 = Sample(
                obs2,
                a2,
                rewards[1],
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                done,
                a2_mem.hidden,
                env_state,
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

            obs, env_state = env.reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
                jnp.zeros((args.num_opps, args.num_envs)),
            ]
            # Player 1
            _a1_mem = agent1.batch_reset(_a1_mem, False)

            # Player 2
            if args.agent1 == "NaiveEx":
                _a1_state, _a1_mem = agent1.batch_init(obs[0])

            if args.agent2 == "NaiveEx":
                _a2_state, _a2_mem = agent2.batch_init(obs[1])

            elif self.args.env_type in ["meta"]:
                # meta-experiments - init 2nd agent per trial
                _a2_state, _a2_mem = agent2.batch_init(
                    jax.random.split(_rng_run, self.num_opps), _a2_mem.hidden
                )
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

            # reset memory
            a1_mem = agent1.batch_reset(a1_mem, False)
            a2_mem = agent2.batch_reset(a2_mem, False)

            # Stats
            if args.env_id == "coin_game":
                env_stats = jax.tree_util.tree_map(
                    lambda x: x,
                    self.cg_stats(env_state),
                )

                rewards_1 = traj_1.rewards.sum(axis=1).mean()
                rewards_2 = traj_2.rewards.sum(axis=1).mean()

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
                rewards_1 = traj_1.rewards.mean()
                rewards_2 = traj_2.rewards.mean()
            else:
                env_stats = {}
                rewards_1 = traj_1.rewards.mean()
                rewards_2 = traj_2.rewards.mean()

            return (
                env_stats,
                rewards_1,
                rewards_2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                a2_metrics,
                traj_1,
                traj_2,
            )

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        agent1, agent2 = agents
        rng, _ = jax.random.split(self.random_key)
        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        wandb.restore(
            name=self.args.model_path1,
            run_path=self.args.run_path1,
            root=os.getcwd(),
        )

        pretrained_params = load(self.args.model_path1)
        a1_state = a1_state._replace(params=pretrained_params)

        if self.args.model_path2:
            print("Loading Second Agent")
            wandb.restore(
                name=self.args.model_path2,
                run_path=self.args.run_path2,
                root=os.getcwd(),
            )
            pretrained_params = load(self.args.model_path2)
            a2_state = a2_state._replace(params=pretrained_params)

        # run actual loop
        rng, rng_run = jax.random.split(rng, 2)
        # RL Rollout
        (
            env_stats,
            rewards_1,
            rewards_2,
            a1_state,
            a1_mem,
            a2_state,
            a2_mem,
            a2_metrics,
            traj,
            other_traj,
        ) = self.rollout(
            rng_run, a1_state, a1_mem, a2_state, a2_mem, env_params
        )

        for stat in env_stats.keys():
            print(stat + f": {env_stats[stat].item()}")
        print(
            f"Total Episode Reward: {float(rewards_1.mean()), float(rewards_2.mean())}"
        )
        print()

        if watchers:
            # metrics [outer_timesteps, num_opps]
            flattened_metrics_2 = jax.tree_util.tree_map(
                lambda x: x.flatten(), a2_metrics
            )
            list_of_metrics = [
                {k: v[i] for k, v in flattened_metrics_2.items()}
                for i in range(len(list(flattened_metrics_2.values())[0]))
            ]
            env_state = traj.env_state
            list_of_env_states = [
                EnvState(
                    red_pos=env_state.red_pos[i, ...],
                    blue_pos=env_state.blue_pos[i, ...],
                    inner_t=env_state.inner_t[i, ...],
                    outer_t=env_state.outer_t[i, ...],
                    grid=env_state.grid[i, ...],
                    red_inventory=env_state.red_inventory[i, ...],
                    blue_inventory=env_state.blue_inventory[i, ...],
                    red_coins=env_state.red_coins[i, ...],
                    blue_coins=env_state.blue_coins[i, ...],
                    freeze=env_state.freeze[i, ...],
                )
                for i in range(self.args.num_outer_steps)
            ]

            list_traj1 = [
                Sample(
                    observations=jax.tree_util.tree_map(
                        lambda x: x[i, ...], traj.observations
                    ),
                    actions=traj.actions[i, ...],
                    rewards=traj.rewards[i, ...],
                    dones=traj.dones[i, ...],
                    env_state=None,
                    behavior_log_probs=traj.behavior_log_probs[i, ...],
                    behavior_values=traj.behavior_values[i, ...],
                    hiddens=traj.hiddens[i, ...],
                )
                for i in range(self.args.num_outer_steps)
            ]

            list_traj2 = [
                Sample(
                    observations=jax.tree_util.tree_map(
                        lambda x: x[i, ...], other_traj.observations
                    ),
                    actions=other_traj.actions[i, ...],
                    rewards=other_traj.rewards[i, ...],
                    dones=other_traj.dones[i, ...],
                    env_state=None,
                    behavior_log_probs=other_traj.behavior_log_probs[i, ...],
                    behavior_values=other_traj.behavior_values[i, ...],
                    hiddens=other_traj.hiddens[i, ...],
                )
                for i in range(self.args.num_outer_steps)
            ]

            list_of_env_stats = [
                jax.tree_util.tree_map(
                    lambda x: x.item(),
                    self.ipditm_stats(
                        env_state,
                        traj_1,
                        traj_2,
                        self.args.num_envs,
                    ),
                )
                for (env_state, traj_1, traj_2) in zip(
                    list_of_env_states, list_traj1, list_traj2
                )
            ]

            # log agent one
            watchers[0](agents[0])

            # log the inner episodes
            for i, metric in enumerate(list_of_metrics):
                agents[1]._logger.metrics = metric
                agents[1]._logger.metrics["sgd_steps"] = i
                watchers[1](agents[1])
                wandb.log({"train_iteration": i} | list_of_env_stats[i])

            wandb.log(
                {
                    "episodes": 1,
                    "eval/meta_reward/player_1": float(
                        rewards_1.mean().item()
                    ),
                    "eval/meta_reward/player_2": float(
                        rewards_2.mean().item()
                    ),
                }
                | jax.tree_util.tree_map(lambda x: x.item(), env_stats),
            )

        if self.args.save_gif:
            print("Generating Renders")
            env = InTheMatrix(
                self.args.num_inner_steps,
                self.num_outer_steps,
                self.args.fixed_coins,
            )
            env_state = traj.env_state
            pics = []

            if self.num_outer_steps == 1:
                pics1 = []
                pics2 = []
            now = datetime.now()

            # reduce timesteps and pick a random env
            env_state = jax.tree_util.tree_map(
                lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                env_state,
            )
            env_idx = jax.random.choice(rng, env_state.red_pos.shape[2])
            opp_idx = jax.random.choice(rng, env_state.red_pos.shape[1])

            env_state = jax.tree_util.tree_map(
                lambda x: x[:, opp_idx, env_idx, ...], env_state
            )
            env_states = [
                EnvState(
                    red_pos=env_state.red_pos[i, ...],
                    blue_pos=env_state.blue_pos[i, ...],
                    inner_t=env_state.inner_t[i, ...],
                    outer_t=env_state.outer_t[i, ...],
                    grid=env_state.grid[i, ...],
                    red_inventory=env_state.red_inventory[i, ...],
                    blue_inventory=env_state.blue_inventory[i, ...],
                    red_coins=env_state.red_coins[i, ...],
                    blue_coins=env_state.blue_coins[i, ...],
                    freeze=env_state.freeze[i, ...],
                )
                for i in range(self.args.num_outer_steps)
            ]
            gif_every_n_eps = 10
            for i, state in enumerate(tqdm(env_states)):
                meta_episode = i // self.args.num_inner_steps
                if (meta_episode % gif_every_n_eps) == 0 or (
                    meta_episode == self.num_outer_steps - 1
                ):
                    img = env.render(state, env_params)
                    pics.append(img)

                    if self.num_outer_steps == 1:
                        img1 = env.render_agent_view(state, agent=0)
                        img2 = env.render_agent_view(state, agent=1)
                        pics1.append(img1)
                        pics2.append(img2)

            print("Saving MP4")
            print(onp.array(pics).shape)
            wandb.log(
                {
                    "video": wandb.Video(
                        onp.transpose(onp.array(pics), (0, 3, 1, 2)),
                        fps=16,
                        format="mp4",
                    )
                }
            )
            print("Saving Gif")
            pics = [Image.fromarray(img) for img in pics]
            print(len(pics))
            pics_by_episode = [
                pics[i : i + self.args.num_inner_steps]
                for i in range(0, len(pics), self.args.num_inner_steps)
            ]
            print(len(pics_by_episode))

            for i, pics in enumerate(pics_by_episode):
                pics[0].save(
                    f"{self.args.wandb.group}_{now}_ep{i * gif_every_n_eps}.gif",
                    format="gif",
                    save_all=True,
                    append_images=pics[1:],
                    duration=100,
                    loop=0,
                    optimize=False,
                )

                wandb.log(
                    {
                        "video": wandb.Video(
                            f"{self.args.wandb.group}_{now}_ep{i * gif_every_n_eps}.gif",
                            fps=4,
                            format="gif",
                        )
                    }
                )

            if self.num_outer_steps == 1:
                pics1 = [Image.fromarray(img) for img in pics1]
                pics2 = [Image.fromarray(img) for img in pics2]

                pics1[0].save(
                    f"{self.args.wandb.group}_agent1_{now}.gif",
                    format="gif",
                    save_all=True,
                    append_images=pics1[1:],
                    duration=300,
                    loop=0,
                    optimize=False,
                )

                pics2[0].save(
                    f"{self.args.wandb.group}_agent2_{now}.gif",
                    format="gif",
                    save_all=True,
                    append_images=pics2[1:],
                    duration=300,
                    loop=0,
                    optimize=False,
                )

                wandb.log(
                    {
                        "video": wandb.Video(
                            f"{self.args.wandb.group}_agent1_{now}.gif",
                            fps=4,
                            format="gif",
                        )
                    }
                )
                wandb.log(
                    {
                        "video": wandb.Video(
                            f"{self.args.wandb.group}_agent2_{now}.gif",
                            fps=4,
                            format="gif",
                        )
                    }
                )

        return agents
