import os
import time
from typing import Any, List, NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import wandb
from pax.utils import MemoryState, TrainingState, copy_state_and_mem, save
from pax.watchers import n_player_ipd_visitation
from pax.watchers.cournot import cournot_stats
from pax.watchers.fishery import fishery_stats

MAX_WANDB_CALLS = 1000


class LOLASample(NamedTuple):
    obs_self: jnp.ndarray
    obs_other: jnp.ndarray
    actions_self: jnp.ndarray
    actions_other: jnp.ndarray
    dones: jnp.ndarray
    rewards_self: jnp.ndarray
    rewards_other: jnp.ndarray


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


class NplayerRLRunner:
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
        self.ipd_stats = n_player_ipd_visitation
        self.cournot_stats = cournot_stats
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
        num_outer_steps = self.args.num_outer_steps
        agent1, *other_agents = agents
        agent1.other_agents = other_agents

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
        if args.agent1 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
            agent1._state, agent1._mem = agent1.batch_init(
                agent1._state.random_key, init_hidden
            )
        if args.agent1 == "LOLA":
            # batch for num_opps
            agent1.batch_in_lookahead = jax.vmap(
                agent1.in_lookahead, (0, None, 0, 0, 0), (0, 0)
            )

        # go through opponents, we start with agent2
        for agent_idx, non_first_agent in enumerate(other_agents):
            agent_arg = f"agent{agent_idx + 2}"
            # equivalent of args.agent_n
            if OmegaConf.select(args, agent_arg) == "NaiveEx":
                # special case where NaiveEx has a different call signature
                non_first_agent.batch_init = jax.jit(
                    jax.vmap(non_first_agent.make_initial_state)
                )
            else:
                non_first_agent.batch_init = jax.vmap(
                    non_first_agent.make_initial_state, (0, None), 0
                )
            non_first_agent.batch_policy = jax.jit(
                jax.vmap(non_first_agent._policy)
            )
            non_first_agent.batch_reset = jax.jit(
                jax.vmap(non_first_agent.reset_memory, (0, None), 0),
                static_argnums=1,
            )
            non_first_agent.batch_update = jax.jit(
                jax.vmap(non_first_agent.update, (1, 0, 0, 0), 0)
            )

            if OmegaConf.select(args, agent_arg) != "NaiveEx":
                # NaiveEx requires env first step to init.
                init_hidden = jnp.tile(
                    non_first_agent._mem.hidden, (args.num_opps, 1, 1)
                )
                agent_rng = jax.random.split(
                    non_first_agent._state.random_key, args.num_opps
                )
                (
                    non_first_agent._state,
                    non_first_agent._mem,
                ) = non_first_agent.batch_init(
                    agent_rng,
                    init_hidden,
                )

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = carry

            # unpack rngs
            rngs = self.split(rngs, 4)
            env_rng = rngs[:, :, 0, :]
            rngs = rngs[:, :, 3, :]
            new_other_agent_mem = [None] * len(other_agents)

            (
                first_action,
                first_agent_state,
                new_first_agent_mem,
            ) = agent1.batch_policy(
                first_agent_state,
                first_agent_obs,
                first_agent_mem,
            )
            actions = [first_action]
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    non_first_action,
                    other_agent_state[agent_idx],
                    new_other_agent_mem[agent_idx],
                ) = non_first_agent.batch_policy(
                    other_agent_state[agent_idx],
                    other_agent_obs[agent_idx],
                    other_agent_mem[agent_idx],
                )
                actions.append(non_first_action)
            (
                all_agent_next_obs,
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                actions,
                env_params,
            )

            first_agent_next_obs, *other_agent_next_obs = all_agent_next_obs
            first_agent_reward, *other_agent_rewards = all_agent_rewards
            if args.agent1 == "MFOS":
                traj1 = MFOSSample(
                    first_agent_obs,
                    first_action,
                    first_agent_reward,
                    new_first_agent_mem.extras["log_probs"],
                    new_first_agent_mem.extras["values"],
                    done,
                    first_agent_mem.hidden,
                    first_agent_mem.th,
                )
            else:
                traj1 = Sample(
                    first_agent_obs,
                    first_action,
                    first_agent_reward,
                    new_first_agent_mem.extras["log_probs"],
                    new_first_agent_mem.extras["values"],
                    done,
                    first_agent_mem.hidden,
                )
            other_traj = [
                Sample(
                    other_agent_obs[agent_idx],
                    actions[agent_idx + 1],
                    other_agent_rewards[agent_idx],
                    new_other_agent_mem[agent_idx].extras["log_probs"],
                    new_other_agent_mem[agent_idx].extras["values"],
                    done,
                    other_agent_mem[agent_idx].hidden,
                )
                for agent_idx in range(len(other_agents))
            ]
            return (
                rngs,
                first_agent_next_obs,
                tuple(other_agent_next_obs),
                first_agent_reward,
                tuple(other_agent_rewards),
                first_agent_state,
                other_agent_state,
                new_first_agent_mem,
                new_other_agent_mem,
                env_state,
                env_params,
            ), (traj1, *other_traj)

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
            )
            other_agent_metrics = [None] * len(other_agents)
            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = vals
            # MFOS has to take a meta-action for each episode
            if args.agent1 == "MFOS":
                first_agent_mem = agent1.meta_policy(first_agent_mem)

            # update second agent
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                    other_agent_metrics[agent_idx],
                ) = non_first_agent.batch_update(
                    trajectories[agent_idx + 1],
                    other_agent_obs[agent_idx],
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                )
            return (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ), (trajectories, other_agent_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            first_agent_state: TrainingState,
            first_agent_mem: MemoryState,
            other_agent_state: List[TrainingState],
            other_agent_mem: List[MemoryState],
            _env_params: Any,
        ):
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
            ] * args.num_players
            # Player 1
            first_agent_mem = agent1.batch_reset(first_agent_mem, False)
            # Other players
            _rng_run, other_agent_rng = jax.random.split(_rng_run, 2)

            # Resetting Agents if necessary
            if args.agent1 == "NaiveEx":
                first_agent_state, first_agent_mem = agent1.batch_init(obs[0])

            for agent_idx, non_first_agent in enumerate(other_agents):
                # indexing starts at 2 for args
                agent_arg = f"agent{agent_idx + 2}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "NaiveEx":
                    (
                        other_agent_mem[agent_idx],
                        other_agent_state[agent_idx],
                    ) = non_first_agent.batch_init(obs[agent_idx + 1])

                elif self.args.env_type in ["meta"]:
                    # meta-experiments - init 2nd agent per trial
                    (
                        other_agent_state[agent_idx],
                        other_agent_mem[agent_idx],
                    ) = non_first_agent.batch_init(
                        jax.random.split(other_agent_rng, self.num_opps),
                        non_first_agent._mem.hidden,
                    )
                    _rng_run, other_agent_rng = jax.random.split(_rng_run, 2)

            # run trials
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (
                    rngs,
                    obs[0],
                    tuple(obs[1:]),
                    rewards[0],
                    tuple(rewards[1:]),
                    first_agent_state,
                    other_agent_state,
                    first_agent_mem,
                    other_agent_mem,
                    env_state,
                    _env_params,
                ),
                None,
                length=num_outer_steps,
            )

            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = vals
            trajectories, other_agent_metrics = stack

            # update outer agent
            if args.agent1 != "LOLA":
                first_agent_state, _, first_agent_metrics = agent1.update(
                    reduce_outer_traj(trajectories[0]),
                    self.reduce_opp_dim(first_agent_obs),
                    first_agent_state,
                    self.reduce_opp_dim(first_agent_mem),
                )

            elif args.agent1 == "LOLA":
                # jax.debug.breakpoint()
                # copy so we don't modify the original during simulation
                first_agent_metrics = None

                self_state, self_mem = copy_state_and_mem(
                    first_agent_state, first_agent_mem
                )
                other_states, other_mems = [], []
                for agent_idx, non_first_agent in enumerate(other_agents):
                    other_state, other_mem = copy_state_and_mem(
                        other_agent_state[agent_idx],
                        other_agent_mem[agent_idx],
                    )
                    other_states.append(other_state)
                    other_mems.append(other_mem)
                # get new state of opponent after their lookahead optimisation
                for _ in range(args.lola.num_lookaheads):
                    _rng_run, _ = jax.random.split(_rng_run)
                    lookahead_rng = jax.random.split(_rng_run, args.num_opps)

                    # we want to batch this num_opps times
                    other_states, other_mems = agent1.batch_in_lookahead(
                        lookahead_rng,
                        self_state,
                        self_mem,
                        other_states,
                        other_mems,
                    )
                # get our new state after our optimisation based on ops new state
                _rng_run, out_look_rng = jax.random.split(_rng_run)
                first_agent_state = agent1.out_lookahead(
                    out_look_rng,
                    first_agent_state,
                    first_agent_mem,
                    other_states,
                    other_mems,
                )
            # jax.debug.breakpoint()

            if args.agent2 == "LOLA":
                raise NotImplementedError("LOLA not implemented for agent2")

            # reset memory
            first_agent_mem = agent1.batch_reset(first_agent_mem, False)
            for agent_idx, non_first_agent in enumerate(other_agents):
                other_agent_mem[agent_idx] = non_first_agent.batch_reset(
                    other_agent_mem[agent_idx], False
                )

            total_rewards = [traj.rewards.mean() for traj in trajectories]
            if args.env_id == "iterated_nplayer_tensor_game":
                total_env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.ipd_stats(
                        trajectories[0].observations,
                        num_players=args.num_players,
                    ),
                )
            elif args.env_id == "Cournot":
                total_env_stats = jax.tree_util.tree_map(
                    lambda x: x,
                    self.cournot_stats(
                        trajectories[0].observations,
                        _env_params,
                        args.num_players,
                    ),
                )
            elif args.env_id == "Fishery":
                total_env_stats = fishery_stats(trajectories, args.num_players)
            else:
                total_env_stats = {}

            return (
                total_env_stats,
                total_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                first_agent_metrics,
                other_agent_metrics,
                trajectories,
            )

        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, num_iters, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        num_iters = max(
            int(num_iters / (self.args.num_envs * self.num_opps)), 1
        )
        log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        save_interval = self.args.save_interval

        agent1, *other_agents = agents
        rng, _ = jax.random.split(self.random_key)

        first_agent_state, first_agent_mem = agent1._state, agent1._mem
        other_agent_mem = [None] * len(other_agents)
        other_agent_state = [None] * len(other_agents)

        for agent_idx, non_first_agent in enumerate(other_agents):
            other_agent_state[agent_idx], other_agent_mem[agent_idx] = (
                non_first_agent._state,
                non_first_agent._mem,
            )

        print(f"Num iters {num_iters}")
        print(f"Log Interval {log_interval}")
        print(f"Save Interval {save_interval}")
        # run actual loop
        for i in range(num_iters):
            rng, rng_run = jax.random.split(rng, 2)
            # RL Rollout
            (
                env_stats,
                total_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                first_agent_metrics,
                other_agent_metrics,
                trajectories,
            ) = self.rollout(
                rng_run,
                first_agent_state,
                first_agent_mem,
                other_agent_state,
                other_agent_mem,
                env_params,
            )

            # saving
            if i % save_interval == 0:
                log_savepath1 = os.path.join(
                    self.save_dir, f"agent1_iteration_{i}"
                )
                if watchers:
                    print(f"Saving iteration {i} locally and to WandB")
                    wandb.save(log_savepath1)
                else:
                    print(f"Saving iteration {i} locally")

            # #logging
            # if i % log_interval == 0:
            #     print(f"Episode {i}")
            #     for stat in env_stats.keys():
            #         print(stat + f": {env_stats[stat].item()}")
            #     print(
            #         f"Reward Per Timestep: {[float(reward.mean()) for reward in total_rewards]}"
            #     )
            #     print()

            if watchers:
                # metrics [outer_timesteps]
                flattened_metrics_1 = jax.tree_util.tree_map(
                    lambda x: jnp.mean(x), first_agent_metrics
                )
                if self.args.agent1 != "LOLA":
                    agent1._logger.metrics = (
                        agent1._logger.metrics | flattened_metrics_1
                    )
                    for agent, metric in zip(
                        other_agents, other_agent_metrics
                    ):
                        flattened_metrics = jax.tree_util.tree_map(
                            lambda x: jnp.mean(x), first_agent_metrics
                        )
                        agent._logger.metrics = (
                            agent._logger.metrics | flattened_metrics
                        )

                for watcher, agent in zip(watchers, agents):
                    watcher(agent)

                env_stats = jax.tree_util.tree_map(
                    lambda x: x.item(), env_stats
                )
                rewards_strs = [
                    "train/reward_per_timestep/player_" + str(i)
                    for i in range(1, len(total_rewards) + 1)
                ]
                rewards_val = [
                    float(reward.mean()) for reward in total_rewards
                ]
                global_welfare = {
                    f"train/global_welfare_per_timestep": float(
                        sum(reward.mean().item() for reward in total_rewards)
                    )
                    / len(total_rewards)
                }

                rewards_dict = dict(zip(rewards_strs, rewards_val))
                wandb_log = (
                    {"train_iteration": i}
                    | rewards_dict
                    | env_stats
                    | global_welfare
                )
                wandb.log(wandb_log)

        agents[0]._state = first_agent_state
        for agent_idx, non_first_agent in enumerate(other_agents):
            agents[agent_idx + 1]._state = other_agent_state[agent_idx]
        return agents
