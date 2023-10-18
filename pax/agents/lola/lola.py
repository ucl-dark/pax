from typing import Any, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from pax import utils
from pax.agents.lola.network import make_network
from pax.runners.runner_marl import Sample
from pax.utils import MemoryState, TrainingState


class LOLASample(NamedTuple):
    obs_self: jnp.ndarray
    obs_other: List[jnp.ndarray]
    actions_self: jnp.ndarray
    actions_other: List[jnp.ndarray]
    dones: jnp.ndarray
    rewards_self: jnp.ndarray
    rewards_other: List[jnp.ndarray]


class Batch(NamedTuple):
    """A batch of data; all shapes are expected to be [B, ...]."""

    observations: jnp.ndarray
    actions: jnp.ndarray
    advantages: jnp.ndarray

    # Target value estimate used to bootstrap the value function.
    target_values: jnp.ndarray

    # Value estimate and action log-prob at behavior time.
    behavior_values: jnp.ndarray
    behavior_log_probs: jnp.ndarray


def magic_box(x):
    return jnp.exp(x - jax.lax.stop_gradient(x))


class Logger:
    metrics: dict


class LOLA:
    """LOLA with the DiCE objective function."""

    def __init__(
        self,
        args,
        network: NamedTuple,
        outer_optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        player_id: int,
        obs_spec: Tuple,
        env_params: Any,
        env_step,
        env_reset,
        num_envs: int = 4,
        num_steps: int = 150,
        use_baseline: bool = True,
        gamma: float = 0.96,
    ):
        self._num_envs = num_envs  # number of environments
        self._num_opps = args.num_opps
        self.env_step = env_step
        self.env_reset = env_reset
        # self.agent2 = None
        self.other_agents = None
        self.args = args

        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            """Agent policy to select actions and calculate agent specific information"""

            key, subkey = jax.random.split(state.random_key)
            dist, values = network.apply(state.params, observation)
            actions = dist.sample(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = dist.log_prob(actions)
            state = state._replace(random_key=key)
            return actions, state, mem

        def outer_loss(params, mem, other_params, other_mems, samples):
            """Used for the outer rollout"""
            # Unpack the samples
            obs_1 = samples.obs_self
            other_obs = samples.obs_other

            # we care about our own rewards
            self_rewards = samples.rewards_self
            actions_1 = samples.actions_self
            other_actions = samples.actions_other
            # jax.debug.breakpoint()

            # Get distribution and value using my network
            distribution, values = self.network.apply(params, obs_1)
            self_log_prob = distribution.log_prob(actions_1)

            # Get distribution and value using other player's network
            other_log_probs = []
            for idx, agent in enumerate(self.other_agents):
                if self.args.agent2 == "PPO_memory":
                    (distribution, _,), _ = agent.network.apply(
                        other_params[idx],
                        other_obs[idx],
                        other_mems[idx].hidden,
                    )
                else:
                    distribution, _ = agent.network.apply(
                        other_params[idx], other_obs[idx]
                    )
                other_log_probs.append(
                    distribution.log_prob(other_actions[idx])
                )

            # flatten opponent and num_envs into one dimension

            # apply discount:
            cum_discount = (
                jnp.cumprod(self.gamma * jnp.ones(self_rewards.shape), axis=0)
                / self.gamma
            )

            discounted_rewards = self_rewards * cum_discount
            discounted_values = values * cum_discount
            # jax.debug.breakpoint()
            # TODO no clue if this makes any sense
            # stochastics nodes involved in rewards dependencies:
            sum_other_log_probs = jnp.sum(
                jnp.stack(other_log_probs, axis=0), axis=0
            )
            dependencies = jnp.cumsum(
                self_log_prob + sum_other_log_probs, axis=0
            )
            # logprob of each stochastic nodes:
            stochastic_nodes = self_log_prob + sum_other_log_probs

            # dice objective:
            dice_objective = jnp.mean(
                jnp.sum(magic_box(dependencies) * discounted_rewards, axis=0)
            )

            if use_baseline:
                # variance_reduction:
                baseline_term = jnp.mean(
                    jnp.sum(
                        (1 - magic_box(stochastic_nodes)) * discounted_values,
                        axis=0,
                    )
                )
                dice_objective = dice_objective + baseline_term

            G_ts = reverse_cumsum(discounted_rewards, axis=0)
            R_ts = G_ts / cum_discount
            # # want to minimize this value
            value_objective = jnp.mean((R_ts - values) ** 2)

            # want to maximize this objective
            loss_total = -dice_objective + value_objective
            return loss_total, {
                "loss_total": -dice_objective + value_objective,
                "loss_policy": -dice_objective,
                "loss_value": value_objective,
            }

        def inner_loss(
            chosen_op_params,
            chosen_op_idx,
            lola_params,
            mem,
            other_params,
            other_mems,
            samples,
        ):
            """Used for the inner rollout"""
            obs_1 = samples.obs_self
            other_obs = samples.obs_other

            # we care about the chosen opponent player's rewards
            self_rewards = samples.rewards_self
            other_rewards = samples.rewards_other
            actions_1 = samples.actions_self
            other_actions = samples.actions_other

            # Get distribution and valwue using my network
            distribution, _ = self.network.apply(lola_params, obs_1)
            self_log_prob = distribution.log_prob(actions_1)

            other_log_probs = []
            other_values = []
            for idx, agent in enumerate(self.other_agents):
                # treating this case separately cause we want the grads on the chosen other players params
                if idx == chosen_op_idx:
                    if self.args.agent2 == "PPO_memory":
                        (
                            distribution,
                            values,
                        ), hidden_state = agent.network.apply(
                            chosen_op_params,
                            other_obs[idx],
                            other_mems[idx].hidden,
                        )

                    else:
                        distribution, values = agent.network.apply(
                            chosen_op_params, other_obs[idx]
                        )
                else:
                    if self.args.agent2 == "PPO_memory":
                        (
                            distribution,
                            values,
                        ), hidden_state = agent.network.apply(
                            other_params[idx],
                            other_obs[idx],
                            other_mems[idx].hidden,
                        )

                    else:
                        distribution, values = agent.network.apply(
                            other_params[idx], other_obs[idx]
                        )
                other_values.append(values)
                other_log_probs.append(
                    distribution.log_prob(other_actions[idx])
                )
            # apply discount:
            cum_discount = (
                jnp.cumprod(self.gamma * jnp.ones(self_rewards.shape), axis=0)
                / self.gamma
            )
            discounted_rewards = [
                reward * cum_discount for reward in other_rewards
            ]
            discounted_values = [
                values * cum_discount for values in other_values
            ]
            # TODO do actual maths here - no idea what this is doing
            # stochastics nodes involved in rewards dependencies:
            # dependencies = jnp.cumsum(self_log_prob + other_log_prob, axis=0)
            # # logprob of each stochastic nodes:
            # stochastic_nodes = self_log_prob + other_log_prob
            sum_other_log_probs = jnp.sum(
                jnp.stack(other_log_probs, axis=0), axis=0
            )
            dependencies = jnp.cumsum(
                self_log_prob + sum_other_log_probs, axis=0
            )
            # logprob of each stochastic nodes:
            stochastic_nodes = self_log_prob + sum_other_log_probs

            # dice objective:
            dice_objective = jnp.mean(
                jnp.sum(
                    magic_box(dependencies)
                    * discounted_rewards[chosen_op_idx],
                    axis=0,
                )
            )

            if use_baseline:
                # variance_reduction:
                baseline_term = jnp.mean(
                    jnp.sum(
                        (1 - magic_box(stochastic_nodes))
                        * discounted_values[chosen_op_idx],
                        axis=0,
                    )
                )
                dice_objective = dice_objective + baseline_term

            G_ts = reverse_cumsum(discounted_rewards[chosen_op_idx], axis=0)
            R_ts = G_ts / cum_discount
            # # want to minimize this value
            value_objective = jnp.mean(
                (R_ts - other_values[chosen_op_idx]) ** 2
            )

            # want to maximize this objective
            loss_total = -dice_objective + value_objective

            return loss_total, {
                "loss_total": -dice_objective + value_objective,
                "loss_policy": -dice_objective,
                "loss_value": value_objective,
            }

        def make_initial_state(
            key: Any, hidden
        ) -> Tuple[TrainingState, MemoryState]:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = outer_optimizer.init(initial_params)
            return TrainingState(
                params=initial_params,
                opt_state=initial_opt_state,
                random_key=key,
                timesteps=0,
            ), MemoryState(
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
                hidden=jnp.zeros((self._num_envs, 1)),
            )

        inner_rollout_rng, random_key = jax.random.split(random_key)
        self.inner_rollout_rng = inner_rollout_rng
        # Initialise training state (parameters, optimiser state, extras).
        self._state, self._mem = make_initial_state(random_key, obs_spec)

        self.make_initial_state = make_initial_state

        # Setup player id
        self.player_id = player_id

        self.env_params = env_params

        grad_inner = jax.grad(inner_loss, has_aux=True, argnums=0)
        grad_outer = jax.grad(outer_loss, has_aux=True, argnums=0)
        # vmap over num_envs
        self.grad_fn_inner = jax.jit(
            jax.vmap(grad_inner, (None, None, None, 0, None, 0, 0), (0, 0)),
            static_argnums=1,
        )
        self.grad_fn_outer = jax.jit(
            jax.vmap(
                jax.vmap(grad_outer, (None, 0, None, 0, 0), (0, 0)),
                (None, 0, 0, 0, 0),
                (0, 0),
            )
        )

        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "loss_total": 0,
            "loss_policy": 0,
            "loss_value": 0,
        }

        # Initialize functions
        self._policy = policy
        self.network = network

        # initialize some variables
        self._outer_optimizer = outer_optimizer
        self.gamma = gamma

        # Other useful hyperparameters
        self._num_steps = num_steps  # number of steps per environment
        self._batch_size = int(num_envs * num_steps)  # number in one batch
        self._obs_spec = obs_spec

    # def select_action(self, state, t: TimeStep):
    #     """Selects action and updates info with PPO specific information"""
    #     actions, state = self._policy(
    #         state.params, t.observation, state
    #     )
    #      utils.to_numpy(actions), state

    def in_lookahead(self, rng, my_state, my_mem, other_states, other_mems):
        """
        Performs a rollout using the current parameters of both agents
        and simulates a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        """

        # do a full rollout
        # we want to play num_envs games at once wiht one opponent
        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, self._num_envs).reshape(
            (self._num_envs, -1)
        )

        batch_reset = jax.vmap(self.env_reset, (0, None), 0)
        obs, env_state = batch_reset(reset_rngs, self.env_params)

        rewards = [jnp.zeros(self._num_envs)] * self.args.num_players

        inner_rollout_rng, rng = jax.random.split(rng)
        inner_rollout_rngs = jax.random.split(
            inner_rollout_rng, self._num_envs
        ).reshape((self._num_envs, -1))
        batch_step = jax.vmap(self.env_step, (0, 0, 0, None), 0)
        batch_policy1 = jax.vmap(self._policy, (None, 0, 0), (0, None, 0))
        batch_policies = [
            jax.vmap(agent._policy, (None, 0, 0), (0, None, 0))
            for agent in self.other_agents
        ]
        # batch_policy2 = jax.vmap(self.agent2._policy, (None, 0, 0), (0, None, 0))

        def lola_inlookahead_rollout(carry, unused):
            """Runner for inner episode"""

            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_states,
                first_agent_mem,
                other_agent_mems,
                env_state,
                env_params,
            ) = carry
            # unpack rngs

            # this fn is not batched over num_envs!
            vmap_split = jax.vmap(jax.random.split, (0, None), 0)
            rngs = vmap_split(rngs, 4)

            env_rng = rngs[:, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, 3, :]

            actions = []
            (
                first_action,
                first_agent_state,
                new_first_agent_mem,
            ) = batch_policy1(
                first_agent_state,
                first_agent_obs,
                first_agent_mem,
            )
            actions.append(first_action)
            new_other_agent_mems = [None] * len(self.other_agents)
            for agent_idx, other_policy in enumerate(batch_policies):
                (
                    non_first_action,
                    other_agent_states[agent_idx],
                    new_other_agent_mems[agent_idx],
                ) = other_policy(
                    other_agent_states[agent_idx],
                    other_agent_obs[agent_idx],
                    other_agent_mems[agent_idx],
                )
                actions.append(non_first_action)

            (
                all_agent_next_obs,
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = batch_step(
                env_rng,
                env_state,
                actions,
                env_params,
            )
            first_agent_next_obs, *other_agent_next_obs = all_agent_next_obs
            first_agent_reward, *other_agent_rewards = all_agent_rewards
            traj1 = LOLASample(
                first_agent_next_obs,
                other_agent_next_obs,
                actions[0],
                actions[1:],
                done,
                first_agent_reward,
                other_agent_rewards,
            )

            other_traj = [
                Sample(
                    other_agent_obs[agent_idx],
                    actions[agent_idx + 1],
                    other_agent_rewards[agent_idx],
                    new_other_agent_mems[agent_idx].extras["log_probs"],
                    new_other_agent_mems[agent_idx].extras["values"],
                    done,
                    other_agent_mems[agent_idx].hidden,
                )
                for agent_idx in range(len(self.other_agents))
            ]
            return (
                rngs,
                first_agent_next_obs,
                tuple(other_agent_next_obs),
                first_agent_reward,
                tuple(other_agent_rewards),
                first_agent_state,
                other_agent_states,
                new_first_agent_mem,
                new_other_agent_mems,
                env_state,
                env_params,
            ), (traj1, *other_traj)

        # jax.debug.breakpoint()

        carry, trajectories = jax.lax.scan(
            lola_inlookahead_rollout,
            (
                inner_rollout_rngs,
                obs[0],
                tuple(obs[1:]),
                rewards[0],
                tuple(rewards[1:]),
                my_state,
                other_states,
                my_mem,
                other_mems,
                env_state,
                self.env_params,
            ),
            None,
            length=self._num_steps,  # num_inner_steps
        )
        my_mem = carry[7]
        other_mems = carry[8]

        # flip axes to get (num_envs, num_inner, obs_dim) to vmap over numenvs
        vmap_trajectories = jax.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), trajectories
        )

        sample = LOLASample(
            obs_self=vmap_trajectories[0].obs_self,
            obs_other=[traj.observations for traj in vmap_trajectories[1:]],
            actions_self=vmap_trajectories[0].actions_self,
            actions_other=[traj.actions for traj in vmap_trajectories[1:]],
            dones=vmap_trajectories[0].dones,
            rewards_self=vmap_trajectories[0].rewards_self,
            rewards_other=[traj.rewards for traj in vmap_trajectories[1:]],
        )

        # get gradients of opponents
        other_gradients = []
        for idx in range(len((self.other_agents))):
            chosen_op_idx = idx
            chosen_op_params = other_states[idx].params

            gradient, _ = self.grad_fn_inner(
                chosen_op_params,
                chosen_op_idx,
                my_state.params,
                my_mem,
                [state.params for state in other_states],
                other_mems,
                sample,
            )
            other_gradients.append(gradient)
        # avg over numenvs
        other_gradients = [
            jax.tree_map(lambda x: x.mean(axis=0), other_gradient)
            for other_gradient in other_gradients
        ]

        # Update the optimizer
        new_other_states = []
        for idx, agent in enumerate(self.other_agents):
            updates, opt_state = agent.optimizer.update(
                other_gradients[idx], other_states[idx].opt_state
            )
            # apply the optimizer updates
            params = optax.apply_updates(other_states[idx].params, updates)

            # replace the other player's current parameters with a simulated update
            new_other_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=other_states[idx].random_key,
                timesteps=other_states[idx].timesteps,
            )
            new_other_states.append(new_other_state)

        return new_other_states, other_mems

    def out_lookahead(self, rng, my_state, my_mem, other_states, other_mems):
        """
        Performs a real rollout using the current parameters of both agents
        and a naive learning update step for the other agent

        INPUT:
        env: SequentialMatrixGame, an environment object of the game being played
        other_agents: list, a list of objects of the other agents
        """

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(
            reset_rng, self._num_envs * self._num_opps
        ).reshape((self._num_opps, self._num_envs, -1))

        batch_reset = jax.vmap(
            jax.vmap(self.env_reset, (0, None), 0), (0, None), 0
        )
        obs, env_state = batch_reset(reset_rngs, self.env_params)

        rewards = [
            jnp.zeros((self._num_opps, self._num_envs)),
        ] * self.args.num_players

        inner_rollout_rng, _ = jax.random.split(rng)
        inner_rollout_rngs = jax.random.split(
            inner_rollout_rng, self._num_envs * self._num_opps
        ).reshape((self._num_opps, self._num_envs, -1))
        batch_step = jax.vmap(
            jax.vmap(self.env_step, (0, 0, 0, None), 0), (0, 0, 0, None), 0
        )

        def lola_outlookahead_rollout(carry, unused):
            """Runner for inner episode"""

            (
                rngs,
                first_agent_obs,
                other_agent_obs,
                first_agent_reward,
                other_agent_rewards,
                first_agent_state,
                other_agent_states,
                first_agent_mem,
                other_agent_mems,
                env_state,
                env_params,
            ) = carry

            # unpack rngs

            vmap_split = jax.vmap(
                jax.vmap(jax.random.split, (0, None), 0), (0, None), 0
            )
            rngs = vmap_split(rngs, 4)

            env_rng = rngs[:, :, 0, :]
            # a1_rng = rngs[:, :, 1, :]
            # a2_rng = rngs[:, :, 2, :]
            rngs = rngs[:, :, 3, :]

            batch_policy1 = jax.vmap(self._policy, (None, 0, 0), (0, None, 0))
            batch_policies = [
                jax.vmap(
                    jax.vmap(agent._policy, (None, 0, 0), (0, None, 0)),
                    (0, 0, 0),
                    (0, 0, 0),
                )
                for agent in self.other_agents
            ]
            actions = []
            (
                first_action,
                first_agent_state,
                new_first_agent_mem,
            ) = batch_policy1(
                first_agent_state,
                first_agent_obs,
                first_agent_mem,
            )
            actions.append(first_action)
            new_other_agent_mems = [None] * len(self.other_agents)
            for agent_idx, other_policy in enumerate(batch_policies):
                (
                    non_first_action,
                    other_agent_states[agent_idx],
                    new_other_agent_mems[agent_idx],
                ) = other_policy(
                    other_agent_states[agent_idx],
                    other_agent_obs[agent_idx],
                    other_agent_mems[agent_idx],
                )
                actions.append(non_first_action)

            (
                all_agent_next_obs,
                env_state,
                all_agent_rewards,
                done,
                info,
            ) = batch_step(
                env_rng,
                env_state,
                actions,
                env_params,
            )

            first_agent_next_obs, *other_agent_next_obs = all_agent_next_obs
            first_agent_reward, *other_agent_rewards = all_agent_rewards
            traj1 = LOLASample(
                first_agent_obs,
                other_agent_obs,
                actions[0],
                actions[1:],
                done,
                first_agent_reward,
                other_agent_rewards,
            )

            other_traj = [
                Sample(
                    other_agent_obs[agent_idx],
                    actions[agent_idx + 1],
                    other_agent_rewards[agent_idx],
                    new_other_agent_mems[agent_idx].extras["log_probs"],
                    new_other_agent_mems[agent_idx].extras["values"],
                    done,
                    other_agent_mems[agent_idx].hidden,
                )
                for agent_idx in range(len(self.other_agents))
            ]
            return (
                rngs,
                first_agent_next_obs,
                tuple(other_agent_next_obs),
                first_agent_reward,
                tuple(other_agent_rewards),
                first_agent_state,
                other_agent_states,
                new_first_agent_mem,
                new_other_agent_mems,
                env_state,
                env_params,
            ), (traj1, *other_traj)

        # do a full rollout
        _, trajectories = jax.lax.scan(
            lola_outlookahead_rollout,
            (
                inner_rollout_rngs,
                obs[0],
                tuple(obs[1:]),
                rewards[0],
                tuple(rewards[1:]),
                my_state,
                other_states,
                my_mem,
                other_mems,
                env_state,
                self.env_params,
            ),
            None,
            length=self._num_steps,
        )
        # num_inner, num_opps, num_envs to num_opps, num_envs, num_inner
        vmap_trajectories = jax.tree_map(
            lambda x: jnp.moveaxis(x, 0, 2), trajectories
        )
        sample = LOLASample(
            obs_self=vmap_trajectories[0].obs_self,
            obs_other=[traj.observations for traj in vmap_trajectories[1:]],
            actions_self=vmap_trajectories[0].actions_self,
            actions_other=[traj.actions for traj in vmap_trajectories[1:]],
            dones=vmap_trajectories[0].dones,
            rewards_self=vmap_trajectories[0].rewards_self,
            rewards_other=[traj.rewards for traj in vmap_trajectories[1:]],
        )
        # print("Before updating")
        # print("---------------------")
        # print("params", self._state.params)
        # print("opt_state", self._state.opt_state)
        # print()
        # calculate the gradients
        gradients, results = self.grad_fn_outer(
            my_state.params,
            my_mem,
            [state.params for state in other_states],
            other_mems,
            sample,
        )
        gradients = jax.tree_map(lambda x: x.mean(axis=(0, 1)), gradients)
        # print("Gradients", gradients)
        # print()
        # Update the optimizer
        updates, opt_state = self._outer_optimizer.update(
            gradients, my_state.opt_state
        )
        # print("Updates", updates)
        # print("Updated optimizer", opt_state)
        # print()

        # apply the optimizer updates
        params = optax.apply_updates(my_state.params, updates)

        # Update internal agent's timesteps
        self._total_steps += self._num_envs
        self._logger.metrics["total_steps"] += self._num_envs
        self._state._replace(timesteps=self._total_steps)

        # Logging
        self._logger.metrics["sgd_steps"] += 1
        self._logger.metrics["loss_total"] = results["loss_total"]
        self._logger.metrics["loss_policy"] = results["loss_policy"]
        self._logger.metrics["loss_value"] = results["loss_value"]

        # replace the player's current parameters with a real update
        new_state = TrainingState(
            params=params,
            opt_state=opt_state,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps,
        )
        return new_state

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
            hidden=jnp.zeros((self._num_envs, 1)),
        )
        return memory


def make_lola(
    args,
    obs_spec,
    action_spec,
    seed: int,
    player_id: int,
    env_params: Any,
    env_step,
    env_reset,
):
    """Make Naive Learner Policy Gradient agent"""
    # Create Haiku network
    network = make_network(action_spec)
    # Outer optimizer uses Adam
    outer_optimizer = optax.adam(args.lola.lr_out)
    # Random key
    random_key = jax.random.PRNGKey(seed=seed)

    return LOLA(
        args=args,
        network=network,
        outer_optimizer=outer_optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        env_params=env_params,
        env_step=env_step,
        env_reset=env_reset,
        player_id=player_id,
        num_envs=args.num_envs,
        num_steps=args.num_inner_steps,
        use_baseline=args.lola.use_baseline,
        gamma=args.lola.gamma,
    )


def reverse_cumsum(x, axis):
    return x + jnp.sum(x, axis=axis, keepdims=True) - jnp.cumsum(x, axis=axis)


if __name__ == "__main__":
    pass
