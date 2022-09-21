from datetime import datetime
import os
import time
from typing import List, NamedTuple

from dm_env import TimeStep
import jax
import jax.numpy as jnp
import wandb

from pax.utils import load, get_advantages
from pax.watchers import cg_visitation, ipd_visitation
from pax.ppo.ppo_gru import Batch


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray
    memory: jnp.ndarray
    state: jnp.ndarray


class EvalRunnerCG:
    """Holds the runner's state."""

    def __init__(self, args):
        self.args = args
        self.num_opps = args.num_opps
        self.eval_steps = 0
        self.eval_episodes = 0
        self.random_key = jax.random.PRNGKey(args.seed)
        self.start_datetime = datetime.now()
        self.start_time = time.time()
        self.train_steps = 0
        self.train_episodes = 0
        self.num_seeds = args.num_seeds
        self.run_path = args.run_path
        self.model_path = args.model_path
        self.ipd_stats = jax.jit(ipd_visitation)
        self.cg_stats = jax.jit(cg_visitation)

    def eval_loop(self, env, agents, num_seeds, watchers):
        """Run evaluation of agents in environment"""

        def _inner_rollout(carry, unused):
            """Runner for inner episode"""
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry

            a1, a1_state, new_a1_mem = agent1.batch_policy(
                a1_state,
                t1.observation,
                a1_mem,
            )

            a2, a2_state, new_a2_mem = agent2.batch_policy(
                a2_state,
                t2.observation,
                a2_mem,
            )

            (tprime_1, tprime_2), env_state = env.batch_step(
                (a1, a2),
                env_state,
            )

            traj1 = Sample(
                t1.observation,
                a1,
                tprime_1.reward,
                new_a1_mem.extras["log_probs"],
                new_a1_mem.extras["values"],
                tprime_1.last(),
                a1_mem.hidden,
                a1_mem,
                a1_state
            )
            traj2 = Sample(
                t2.observation,
                a2,
                tprime_2.reward,
                new_a2_mem.extras["log_probs"],
                new_a2_mem.extras["values"],
                tprime_2.last(),
                a2_mem.hidden,
                a2_mem,
                a2_state
            )
            return (
                tprime_1,
                tprime_2,
                a1_state,
                new_a1_mem,
                a2_state,
                new_a2_mem,
                env_state,
            ), (
                traj1,
                traj2,
            )

        def _outer_rollout(carry, unused):
            """Runner for trial"""
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = carry
            # play episode of the game
            vals, trajectories = jax.lax.scan(
                _inner_rollout,
                carry,
                None,
                length=env.inner_episode_length,
            )

            # update second agent
            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals

            # do second agent update
            final_t2 = t2._replace(
                step_type=2 * jnp.ones_like(vals[1].step_type)
            )
            a2_state, a2_mem, a2_metrics = agent2.batch_update(
                trajectories[1], final_t2, a2_state, a2_mem
            )

            return (
                t1,
                t2,
                a1_state,
                a1_mem,
                a2_state,
                a2_mem,
                env_state,
            ), (*trajectories, a2_metrics, final_t2)

        print("Evaluation")
        print("------------------------------")
        print(f"Number of Seeds: {self.num_seeds}")
        print(f"Number of Environments: {env.num_envs}")
        print(f"Number of Opponent: {self.num_opps}")
        print("-----------------------")
        # Initialize agents and RNG
        agent1, agent2 = agents.agents
        rng, _ = jax.random.split(self.random_key)

        # Load model
        print("Local model_path", os.path.join(os.getcwd(), self.model_path))
        if watchers:
            wandb.restore(
                name=self.model_path, run_path=self.run_path, root=os.getcwd()
            )
        params = load(self.model_path)

        a1_state, a1_mem = agent1._state, agent1._mem
        a2_state, a2_mem = agent2._state, agent2._mem

        a1_state = a1_state._replace(params=params)

        mean_rewards_p1 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_rewards_p2 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_coop_prob_p1 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_coop_prob_p2 = jnp.zeros(shape=(num_seeds, env.num_trials))
        mean_coins_per_episode_p1 = jnp.zeros(
            shape=(num_seeds, env.num_trials)
        )
        mean_coins_per_episode_p2 = jnp.zeros(
            shape=(num_seeds, env.num_trials)
        )
        for i in range(self.num_seeds):
            rng, rng_run = jax.random.split(rng)
            t_init, env_state = env.runner_reset(
                (self.num_opps, env.num_envs), rng_run
            )

            if self.args.agent2 == "NaiveEx":
                a2_state, a2_mem = agent2.batch_init(t_init[1])

            elif (
                self.args.env_type in ["meta", "infinite"]
                or self.args.coin_type == "coin_meta"
            ):
                # meta-experiments - init 2nd agent per trial
                a2_state, a2_mem = agent2.batch_init(
                    jax.random.split(rng, self.num_opps), a2_mem.hidden
                )
            # run trials
            # len: 16
            # a1: -> / a2: <- / a3: -> ... end of episode: grad:-><-
            vals, stack = jax.lax.scan(
                _outer_rollout,
                (*t_init, a1_state, a1_mem, a2_state, a2_mem, env_state),
                None,
                length=env.num_trials,
            )

            t1, t2, a1_state, a1_mem, a2_state, a2_mem, env_state = vals
            traj_1, traj_2, a2_metrics, final_t2 = stack

            if self.args.env_type == "coin_game":
                env_stats = self.cg_stats(env_state)
                rewards_0 = traj_1.rewards.sum(axis=1).mean()
                rewards_1 = traj_2.rewards.sum(axis=1).mean()

                mean_rewards_p1 = mean_rewards_p1

                mean_coop_prob_p1 = mean_coop_prob_p1.at[i, :].set(
                    env_stats["prob_coop/1"]
                )
                mean_coop_prob_p2 = mean_coop_prob_p2.at[i, :].set(
                    env_stats["prob_coop/2"]
                )
                mean_coins_per_episode_p1 = mean_coins_per_episode_p1.at[
                    i, :
                ].set(env_stats["coins_per_episode/1"])
                mean_coins_per_episode_p2 = mean_coins_per_episode_p2.at[
                    i, :
                ].set(env_stats["coins_per_episode/2"])

            elif self.args.env_type in [
                "meta",
                "sequential",
            ]:
                final_t1 = t1._replace(
                    step_type=2 * jnp.ones_like(t1.step_type)
                )

                env_stats = jax.tree_util.tree_map(
                    lambda x: x.item(), self.ipd_stats(traj_1, final_t1)
                )
                rewards_0 = traj_1.rewards.mean()
                rewards_1 = traj_2.rewards.mean()
            else:
                env_stats = {}

            print(f"Iteration: {i}")
            print(
                "--------------------------------------------------------------------------"
            )
            print(f"Episode Reward: {float(rewards_0), float(rewards_1)}")
            print(
                f"Cooperation Probability: {mean_coop_prob_p1[i].mean()}, {mean_coop_prob_p2[i].mean()}"
            )
            mean_coin_per_ep_p1 = mean_coins_per_episode_p1[i].mean()
            mean_coin_per_ep_p2 = mean_coins_per_episode_p2[i].mean()
            print(
                f"Coins per Episode: {mean_coin_per_ep_p1}, {mean_coin_per_ep_p2}"
            )
            print(
                "--------------------------------------------------------------------------"
            )
            print(
                "--------------------------------------------------------------------------"
            )

            # trial rewards
            for out_step in range(env.num_trials):
                rewards_trial_mean_p1 = (
                    traj_1.rewards[out_step].sum(axis=0).mean()
                )
                rewards_trial_mean_p2 = (
                    traj_2.rewards[out_step].sum(axis=0).mean()
                )
                mean_rewards_p1 = mean_rewards_p1.at[i, out_step].set(
                    rewards_trial_mean_p1
                )
                mean_rewards_p2 = mean_rewards_p2.at[i, out_step].set(
                    rewards_trial_mean_p2
                )
                if out_step % 100 == 0:
                    print(f"Trial {out_step}")
                    print(
                        f"Reward | P1:{rewards_trial_mean_p1}, P2:{rewards_trial_mean_p2}"
                    )
                    coop_prob_p1 = mean_coop_prob_p1[i, out_step]
                    coop_prob_p2 = mean_coop_prob_p2[i, out_step]
                    print(
                        f"Probability of Cooperation | P1:{coop_prob_p1}, P2:{coop_prob_p2}"
                    )
                    coin_per_ep_p1 = mean_coins_per_episode_p1[i, out_step]
                    coin_per_ep_p2 = mean_coins_per_episode_p2[i, out_step]
                    print(
                        f"Coins per Episode | P1:{coin_per_ep_p1}, P2:{coin_per_ep_p2}"
                    )
                    print()

                if watchers:
                    eval_trial_log = {
                        "eval/trial": out_step + 1,
                        f"eval/reward_trial_p1_opp_{i}": rewards_trial_mean_p1,
                        f"eval/reward_trial_p2_opp_{i}": rewards_trial_mean_p2,
                        f"eval/reward_trial_p1_prob_{i}": coop_prob_p1,
                        f"eval/reward_trial_p2_prob_{i}": coop_prob_p2,
                        f"eval/coin_per_ep_p1_{i}": coin_per_ep_p1,
                        f"eval/coin_per_ep_p2_{i}": coin_per_ep_p2,
                    }
                    wandb.log(eval_trial_log)
                # TODO: Add step rewards?

            if watchers:
                wandb_log = {
                    "eval/time/minutes": float(
                        (time.time() - self.start_time) / 60
                    ),
                    "eval/time/seconds": float(
                        (time.time() - self.start_time)
                    ),
                    "eval/episode_reward/player_1": rewards_0,
                    "eval/episode_reward/player_2": rewards_1,
                }
                # wandb_log.update(env_stats)
                # print(env_stats)
                # print(wandb_log)
                wandb_log = wandb_log | env_stats

                # player 2 metrics
                # metrics [outer_timesteps, num_opps]
                flattened_metrics = jax.tree_util.tree_map(
                    lambda x: jnp.sum(jnp.mean(x, 1)), a2_metrics
                )

                agent2._logger.metrics = (
                    agent2._logger.metrics | flattened_metrics
                )

                agent1._logger.metrics = (
                    agent1._logger.metrics | flattened_metrics
                )
                agents.log(watchers)
                # wandb.log(wandb_log)
            print()

        # python -m pax.experiment +experiment/cg=earl_ppo_memory ++eval=True ++num_seeds=1
        # ++num_envs=1 ++num_steps=24 ++num_inner_steps=8 ++ppo.num_minibatches=2
        # Shape:
        # (num_episodes, num_opponents, num_epochs, num_minibatches, hidden_size, output_size)

        # Keys:
        # ['categorical_value_head/~/linear',
        # 'categorical_value_head/~/linear_1',
        # 'gru', 'mlp/~/linear_0',
        # 'mlp/~/linear_1']

        # # take mean over axis=[1,2,3] -> (num_episodes, hidden size, output size)
        # Layer name: categorical_value_head/~/linear
        # Key name: w
        # #                         mean          ravel
        # Shape: (3, 1, 2, 4, 16, 5) -> (3, 16, 5)  ->  (3 * 16 * 5, )

        # Layer name: categorical_value_head/~/linear_1
        # Key name: w
        # Shape: (3, 1, 2, 4, 16, 1) -> (3, 16, 1)

        # Layer name: gru
        # Key name: b
        # Shape: (3, 1, 2, 4, 48) -> (3, 48)
        # Key name: w_h
        # Shape: (3, 1, 2, 4, 16, 48) -> (3, 16, 48)
        # Key name: w_i
        # Shape: (3, 1, 2, 4, 16, 48) -> (3, 16, 48)

        # Layer name: mlp/~/linear_0
        # Key name: b
        # Shape: (3, 1, 2, 4, 16) -> (3, 16)
        # Key name: w
        # Shape: (3, 1, 2, 4, 36, 16) -> (3, 36, 16)

        # Layer name: mlp/~/linear_1
        # Key name: b
        # Shape: (3, 1, 2, 4, 16) -> (3, 16)
        # Key name: w
        # Shape: (3, 1, 2, 4, 16, 16) -> (3, 16, 16)

        # def pytree_layer_shapes(dict):
        #     for layer in grads.keys():
        #         print(f"Layer name: {layer}")
        #         for key in grads[layer].keys():
        #             print(f"Key name: {key}")
        #             print(f"Shape: {grads[layer][key].shape}")
        #         print()
        #     return
        # num_envs = traj_2.observations.shape[2] * traj_2.observations.shape[3]
        # traj_2 = jax.tree_util.tree_map(
        #     lambda x: x.reshape(x.shape[:2] + (num_envs,) + x.shape[4:]),
        #     traj_2,
        # )

        eps_idx = 1
        chunk_size = 2
        final_a2_state = jax.tree_map(lambda x: x[eps_idx][-1], traj_2.state)# traj_2.state[eps_idx]
        a2_mem = jax.tree_map(lambda x: x[eps_idx][-1], traj_2.memory) # traj_2.memory[eps_idx]
        timestep = jax.tree_map(lambda x: x[eps_idx], final_t2) #final_t2[eps_idx]
        traj_2 = jax.tree_map(lambda x: x[eps_idx], traj_2)
        # import pdb; pdb.set_trace()

        _, _, mem = agent2.batch_policy(final_a2_state, timestep.observation, a2_mem)
        prepared_traj_batch = agent2.prepare_batch(traj_2, timestep, mem.extras)

        rewards = prepared_traj_batch.rewards
        behavior_values = prepared_traj_batch.behavior_values
        dones = prepared_traj_batch.dones
        advantages, target_values = agent2.gae_advantages(
        rewards=rewards, values=behavior_values, dones=dones
        )

        intra_bins = []
        behavior_values = behavior_values[:-1, :]
        for t in range(0, 16, chunk_size):
            trajectories = Batch(
                observations=prepared_traj_batch.observations[t:t+chunk_size],
                actions=prepared_traj_batch.actions[t:t+chunk_size],
                advantages=advantages[t:t+chunk_size],
                behavior_log_probs=prepared_traj_batch.behavior_log_probs[t:t+chunk_size],
                target_values=target_values[t:t+chunk_size],
                behavior_values=behavior_values[t:t+chunk_size],
                hiddens=prepared_traj_batch.hiddens[t:t+chunk_size],
            )
            # Concatenate all trajectories. Reshape from [num_envs, num_steps, ..]
            # to [num_envs * num_steps,..]
            assert len(target_values.shape) > 1
            num_envs = target_values.shape[1]
            num_steps = target_values[t:t+chunk_size].shape[0]
            batch_size = num_envs * num_steps

            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )
            params = jax.tree_map(lambda x: x[eps_idx][t], traj_2.state).params# traj_2.state[eps_idx]

            advantages_ = (
                batch.advantages - jnp.mean(batch.advantages, axis=0)
            ) / (jnp.std(batch.advantages, axis=0) + 1e-8)
            gradients, metrics = agent2.grad_fn(
                params,
                0,
                batch.observations[:,0,:],
                batch.actions,
                batch.behavior_log_probs,
                batch.target_values,
                advantages_,
                batch.behavior_values,
                batch.hiddens,
            )
            gradients, _ =jax.flatten_util.ravel_pytree(gradients)
            intra_bins.append(gradients)

        print("Bins", len(intra_bins))
        grid = jnp.zeros(shape=(8, 8))
        for i in range(8):
            grad_i = intra_bins[i]
            for j in range(8):
                grad_j = intra_bins[j]
                cos_sim = jnp.dot(grad_i, grad_j) / (
                    jnp.sqrt(jnp.dot(grad_i, grad_i) * jnp.dot(grad_j, grad_j))
                )
                grid = grid.at[i, j].set(cos_sim)
        print('Intraepisodal', repr(grid))            
        #   state, mem, metrics = self._sgd_step(state, prepared_traj_batch)
        # 1. s1, a1, s2, a2, ..., s100, a100 <- inner episode
        # 2. Calculate advantages (calling update (which calls prepare batch and sgd (which calls gae)))
        # 3. get_grad(loss, [s1, a1, s2, a2, ..., s10, a10])
        # Super janky viz #
        num_bins = 10
        bin_size = int(env.num_trials / num_bins)  # 600 / 10
        bins = []
        grid = jnp.zeros(shape=(10, 10))
        grads = jax.tree_map(
            lambda x: jnp.mean(x, axis=[1, 2, 3]), a2_metrics["gradients"]
        )  # mean over minibatches and epochs

        print("Number of bins:", num_bins)
        print("Bin size:", bin_size)
        print("Num trials", env.num_trials)

        for i in range(0, env.num_trials, bin_size):  # 0, 600, 60

            # flake8: noqa: C901
            bin_i_params = jax.tree_map(
                lambda x: jnp.mean(x[i : i + bin_size], axis=0), grads
            )
            bin_i_params, _ = jax.flatten_util.ravel_pytree(bin_i_params)
            bins.append(bin_i_params)

        print("Bins", len(bins))
        for i in range(10):
            grad_i = bins[i]
            for j in range(10):
                grad_j = bins[j]
                cos_sim = jnp.dot(grad_i, grad_j) / (
                    jnp.sqrt(jnp.dot(grad_i, grad_i) * jnp.dot(grad_j, grad_j))
                )
                grid = grid.at[i, j].set(cos_sim)
        print(repr(grid))
        # super janky viz #

        for out_step in range(env.num_trials):
            if watchers:
                wandb.log(
                    {
                        "eval/trial": out_step + 1,
                        "eval/reward/p1": mean_rewards_p1[:, out_step].mean(),
                        "eval/reward/p2": mean_rewards_p2[:, out_step].mean(),
                        "eval/prob_cooperation/p1": mean_coop_prob_p1[
                            :, out_step
                        ].mean(),
                        "eval/prob_cooperation/p2": mean_coop_prob_p2[
                            :, out_step
                        ].mean(),
                        "eval/coins_per_episode/p1": mean_coins_per_episode_p1[
                            :, out_step
                        ].mean(),
                        "eval/coins_per_episode/p2": mean_coins_per_episode_p2[
                            :, out_step
                        ].mean(),
                    }
                )

        return agents
