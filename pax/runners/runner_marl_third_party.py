import os
import time
from typing import Any, List, NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

import wandb
from pax.utils import MemoryState, TrainingState, save
from pax.watchers import third_party_random_visitation

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


class ThirdPartyRLRunner:
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
        self.third_party_punishment_stats = jax.jit(
            third_party_random_visitation
        )
        # VMAP for num envs: we vmap over the rng but not params
        env.reset = jax.vmap(env.reset, (0, None), 0)
        env.step = jax.vmap(
            env.step, (0, 0, 0, 0, 0, None), 0  # rng, state, actions, params
        )

        # VMAP for num opps: we vmap over the rng but not params
        env.reset = jax.jit(jax.vmap(env.reset, (0, None), 0))
        env.step = jax.jit(
            jax.vmap(
                env.step,
                (0, 0, 0, 0, 0, None),
                0,  # rng, state, actions, params
            )
        )

        self.split = jax.vmap(jax.vmap(jax.random.split, (0, None)), (0, None))
        num_outer_steps = self.args.num_outer_steps
        agent1, *other_agents = agents

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

        # go through opponents, we start with agent2
        for agent_idx, non_first_agent in enumerate(other_agents):
            agent_arg = f"agent{agent_idx+2}"
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
                prev_actions,
                prev_player_selection,
                obs,
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
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]
            new_other_agent_mem = [None] * len(other_agents)
            actions = []

            (
                first_action,
                first_agent_state,
                new_first_agent_mem,
            ) = agent1.batch_policy(
                first_agent_state,
                obs,
                first_agent_mem,
            )
            actions.append(first_action)
            for agent_idx, non_first_agent in enumerate(other_agents):
                (
                    non_first_action,
                    other_agent_state[agent_idx],
                    new_other_agent_mem[agent_idx],
                ) = non_first_agent.batch_policy(
                    other_agent_state[agent_idx],
                    obs,
                    other_agent_mem[agent_idx],
                )
                actions.append(non_first_action)
            (
                player_selection,
                (next_obs,log_obs),
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = env.step(
                env_rng,
                env_state,
                prev_actions,
                prev_player_selection,
                actions,
                env_params,
            )
            first_agent_reward, *other_agent_rewards = all_agent_rewards
            if args.agent1 == "MFOS":
                traj1 = MFOSSample(
                    obs,
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
                    obs,
                    first_action,
                    first_agent_reward,
                    new_first_agent_mem.extras["log_probs"],
                    new_first_agent_mem.extras["values"],
                    done,
                    first_agent_mem.hidden,
                )
            other_traj = [
                Sample(
                    obs,
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
                actions,  # this are the next prev actions
                player_selection,  # this is the player selection for above actions
                next_obs,
                first_agent_reward,
                tuple(other_agent_rewards),
                first_agent_state,
                other_agent_state,
                new_first_agent_mem,
                new_other_agent_mem,
                env_state,
                env_params,
            ), (log_obs, traj1, *other_traj)

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            # play episode of the game
            vals, stack = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=self.args.num_inner_steps,
            )
            log_obs = stack[0]
            trajectories = stack[1:]
            other_agent_metrics = [None] * len(other_agents)
            (
                rngs,
                _1,
                _2,
                obs,
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
                    obs,
                    other_agent_state[agent_idx],
                    other_agent_mem[agent_idx],
                )
            return (
                rngs,
                _1,
                _2,
                obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ), (log_obs,trajectories, other_agent_metrics)

        def _rollout(
            _rng_run: jnp.ndarray,
            first_agent_state: TrainingState,
            first_agent_mem: MemoryState,
            other_agent_state: List[TrainingState],
            other_agent_mem: List[MemoryState],
            _env_params: Any,
        ):
            (
                _rng_run,
                first_action_rng1,
                first_action_rng2,
                first_action_rng3,
            ) = jax.random.split(_rng_run, 4)
            # env reset
            rngs = jnp.concatenate(
                [jax.random.split(_rng_run, args.num_envs)] * args.num_opps
            ).reshape((args.num_opps, args.num_envs, -1))

            obs, env_state = env.reset(rngs, _env_params)
            rewards = [
                jnp.zeros((args.num_opps, args.num_envs)),
            ] * args.num_players
            # first action is uniform random C or D with no punishment - so action 0 or 4
            first_action1 = (
                jax.random.randint(
                    first_action_rng1,
                    (args.num_opps, args.num_envs),
                    0,
                    2,                
                )
                * 4
            )
            first_action2 = (
                jax.random.randint(
                    first_action_rng2,
                    (args.num_opps, args.num_envs),
                    0,
                    2,
                )
                * 4
            )
            first_action3 = (
                jax.random.randint(
                    first_action_rng3,
                    (args.num_opps, args.num_envs),
                    0,
                    2,
             )
                * 4
            )
            first_action = [first_action1, first_action2, first_action3]
            _rng_run, player_sel_rng = jax.random.split(
                _rng_run,
            )

            # first_player_selections is indep permutation of arange(0,3) with shape ( num_opps, num_envs,3
            player_ids = jnp.tile(
                jnp.arange(3), (args.num_opps, args.num_envs, 1)
            )
            first_player_selections = jax.random.permutation(
                key=player_sel_rng, x=player_ids, axis=-1, independent=True,
            )

            # Player 1
            first_agent_mem = agent1.batch_reset(first_agent_mem, False)
            # Other players
            _rng_run, other_agent_rng = jax.random.split(_rng_run, 2)

            # Resetting Agents if necessary
            if args.agent1 == "NaiveEx":
                first_agent_state, first_agent_mem = agent1.batch_init(
                    obs,
                )

            for agent_idx, non_first_agent in enumerate(other_agents):
                # indexing starts at 2 for args
                agent_arg = f"agent{agent_idx+2}"
                # equivalent of args.agent_n
                if OmegaConf.select(args, agent_arg) == "NaiveEx":
                    (
                        other_agent_mem[agent_idx],
                        other_agent_state[agent_idx],
                    ) = non_first_agent.batch_init(
                        obs,
                    )

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
                    first_action,
                    first_player_selections,
                    obs,
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
                _1,
                _2,
                obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_state,
                first_agent_mem,
                other_agent_mem,
                env_state,
                env_params,
            ) = vals
            log_obs, trajectories, other_agent_metrics = stack
            # update outer agent
            first_agent_state, _, first_agent_metrics = agent1.update(
                reduce_outer_traj(trajectories[0]),
                self.reduce_opp_dim(obs),
                first_agent_state,
                self.reduce_opp_dim(first_agent_mem),
            )

            # reset memory
            first_agent_mem = agent1.batch_reset(first_agent_mem, False)
            for agent_idx, non_first_agent in enumerate(other_agents):
                other_agent_mem[agent_idx] = non_first_agent.batch_reset(
                    other_agent_mem[agent_idx], False
                )
            # Stats
            if args.env_id in [
                "third_party_random",
            ]:
                total_env_stats = jax.tree_util.tree_map(
                    lambda x: x.mean(),
                    self.third_party_punishment_stats(
                        log_obs,
                    ),
                )
                total_rewards = [traj.rewards.mean() for traj in trajectories]
            else:
                total_env_stats = {}
                total_rewards = [traj.rewards.mean() for traj in trajectories]

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

        # self.rollout = _rollout
        self.rollout = jax.jit(_rollout)

    def run_loop(self, env_params, agents, num_iters, watchers):
        """Run training of agents in environment"""
        print("Training")
        print("-----------------------")
        # log_interval = int(max(num_iters / MAX_WANDB_CALLS, 5))
        log_interval = 100
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
                agent1._logger.metrics = (
                    agent1._logger.metrics | flattened_metrics_1
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
