# Adapted from https://github.com/deepmind/acme/blob/master/acme/agents/jax/ppo/learning.py

from typing import Any, Dict, NamedTuple, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from pax import utils
from pax.agents.agent import AgentInterface
from pax.agents.ppo.networks import (
    make_coingame_network,
    make_ipditm_network,
    make_sarl_network,
    make_cournot_network,
    make_fishery_network,
    make_rice_sarl_network,
    make_ipd_network,
)
from pax.envs.iterated_matrix_game import IteratedMatrixGame
from pax.envs.iterated_tensor_game_n_player import IteratedTensorGameNPlayer
from pax.envs.rice.c_rice import ClubRice
from pax.envs.rice.rice import Rice
from pax.envs.rice.sarl_rice import SarlRice
from pax.utils import (
    Logger,
    MemoryState,
    TrainingState,
    get_advantages,
    float_precision,
)


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


class PPO(AgentInterface):
    """A simple PPO agent using JAX"""

    def __init__(
        self,
        network: NamedTuple,
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: Tuple,
        num_envs: int = 4,
        num_minibatches: int = 16,
        num_epochs: int = 4,
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3_000_000,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        tabular: bool = False,
        player_id: int = 0,
    ):
        @jax.jit
        def policy(
            state: TrainingState, observation: jnp.ndarray, mem: MemoryState
        ):
            """Agent policy to select actions and calculate agent specific information"""
            key, subkey = jax.random.split(state.random_key)
            dist, values = network.apply(state.params, observation)
            # Calculating logprob separately can cause numerical issues
            # https://github.com/deepmind/distrax/issues/7
            actions, log_prob = dist.sample_and_log_prob(seed=subkey)
            mem.extras["values"] = values
            mem.extras["log_probs"] = log_prob
            mem = mem._replace(extras=mem.extras)
            state = state._replace(random_key=key)
            return actions, state, mem

        @jax.jit
        def gae_advantages(
            rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculates the gae advantages from a sequence. Note that the
            arguments are of length = rollout length + 1"""
            # 'Zero out' the terminated states
            discounts = gamma * jnp.logical_not(dones)

            reverse_batch = (
                jnp.flip(values[:-1], axis=0),
                jnp.flip(rewards, axis=0),
                jnp.flip(discounts, axis=0),
            )

            _, advantages = jax.lax.scan(
                get_advantages,
                (
                    jnp.zeros_like(values[-1]),
                    values[-1],
                    jnp.ones_like(values[-1]) * gae_lambda,
                ),
                reverse_batch,
            )

            advantages = jnp.flip(advantages, axis=0)
            target_values = values[:-1] + advantages  # Q-value estimates
            target_values = jax.lax.stop_gradient(target_values)
            return advantages, target_values

        def loss(
            params: hk.Params,
            timesteps: int,
            observations: jnp.ndarray,
            actions: jnp.array,
            behavior_log_probs: jnp.array,
            target_values: jnp.array,
            advantages: jnp.array,
            behavior_values: jnp.array,
        ):
            """Surrogate loss using clipped probability ratios."""
            distribution, values = network.apply(params, observations)
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy()

            # Compute importance sampling weights: current policy / behavior policy.
            rhos = jnp.exp(log_prob - behavior_log_probs)

            # Policy loss: Clipping
            clipped_ratios_t = jnp.clip(
                rhos, 1.0 - ppo_clipping_epsilon, 1.0 + ppo_clipping_epsilon
            )
            clipped_objective = jnp.fmin(
                rhos * advantages, clipped_ratios_t * advantages
            )
            policy_loss = -jnp.mean(clipped_objective)

            # Value loss: MSE
            value_cost = value_coeff
            unclipped_value_error = target_values - values
            unclipped_value_loss = unclipped_value_error**2

            # Value clipping
            if clip_value:
                # Clip values to reduce variablility during critic training.
                clipped_values = behavior_values + jnp.clip(
                    values - behavior_values,
                    -ppo_clipping_epsilon,
                    ppo_clipping_epsilon,
                )
                clipped_value_error = target_values - clipped_values
                clipped_value_loss = clipped_value_error**2
                value_loss = jnp.mean(
                    jnp.fmax(unclipped_value_loss, clipped_value_loss)
                )
            else:
                value_loss = jnp.mean(unclipped_value_loss)

            # Entropy loss: Standard entropy term
            # Calculate the new value based on linear annealing formula
            if anneal_entropy:
                fraction = jnp.fmax(1 - timesteps / entropy_coeff_horizon, 0)
                entropy_cost = (
                    fraction * entropy_coeff_start
                    + (1 - fraction) * entropy_coeff_end
                )
            # Constant Entropy term
            else:
                entropy_cost = entropy_coeff_start
            entropy_loss = -jnp.mean(entropy)

            # Total loss: Minimize policy and value loss; maximize entropy
            total_loss = (
                policy_loss
                + entropy_cost * entropy_loss
                + value_loss * value_cost
            )

            return total_loss, {
                "loss_total": total_loss,
                "loss_policy": policy_loss,
                "loss_value": value_loss,
                "loss_entropy": entropy_loss,
                "entropy_cost": entropy_cost,
            }

        @jax.jit
        def sgd_step(
            state: TrainingState, sample: NamedTuple
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """Performs a minibatch SGD step, returning new state and metrics."""

            # Extract data
            (
                observations,
                actions,
                rewards,
                behavior_log_probs,
                behavior_values,
                dones,
            ) = (
                sample.observations,
                sample.actions,
                sample.rewards,
                sample.behavior_log_probs,
                sample.behavior_values,
                sample.dones,
            )

            advantages, target_values = gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_steps, num_envs, ..]
            behavior_values = behavior_values[:-1, :]
            trajectories = Batch(
                observations=observations,
                actions=actions,
                advantages=advantages,
                behavior_log_probs=behavior_log_probs,
                target_values=target_values,
                behavior_values=behavior_values,
            )

            # Concatenate all trajectories. Reshape from [num_envs, num_steps, ..]
            # to [num_envs * num_steps,..]
            assert len(target_values.shape) > 1
            num_envs = target_values.shape[1]
            num_steps = target_values.shape[0]
            batch_size = num_envs * num_steps
            assert batch_size % num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            # Compute gradients.
            grad_fn = jax.jit(jax.grad(loss, has_aux=True))

            @jax.jit
            def model_update_minibatch(
                carry: Tuple[hk.Params, optax.OptState, int],
                minibatch: Batch,
            ) -> Tuple[
                Tuple[hk.Params, optax.OptState, int], Dict[str, jnp.ndarray]
            ]:
                """Performs model update for a single minibatch."""
                params, opt_state, timesteps = carry
                # Normalize advantages at the minibatch level before using them.
                advantages = (
                    minibatch.advantages
                    - jnp.mean(minibatch.advantages, axis=0)
                ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
                gradients, metrics = grad_fn(
                    params,
                    timesteps,
                    minibatch.observations,
                    minibatch.actions,
                    minibatch.behavior_log_probs,
                    minibatch.target_values,
                    advantages,
                    minibatch.behavior_values,
                )

                # Apply updates
                updates, opt_state = optimizer.update(gradients, opt_state)
                params = optax.apply_updates(params, updates)

                metrics["norm_grad"] = optax.global_norm(gradients)
                metrics["norm_updates"] = optax.global_norm(updates)
                return (params, opt_state, timesteps), metrics

            @jax.jit
            def model_update_epoch(
                carry: Tuple[
                    jnp.ndarray, hk.Params, optax.OptState, int, Batch
                ],
                unused_t: Tuple[()],
            ) -> Tuple[
                Tuple[jnp.ndarray, hk.Params, optax.OptState, Batch],
                Dict[str, jnp.ndarray],
            ]:
                """Performs model updates based on one epoch of data."""
                key, params, opt_state, timesteps, batch = carry
                key, subkey = jax.random.split(key)
                permutation = jax.random.permutation(subkey, batch_size)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                (params, opt_state, timesteps), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state, timesteps),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, timesteps, batch), metrics

            params = state.params
            opt_state = state.opt_state
            timesteps = state.timesteps

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            # signature is scan(function, carry, tuple to iterate over, length)
            (key, params, opt_state, timesteps, _), metrics = jax.lax.scan(
                model_update_epoch,
                (state.random_key, params, opt_state, timesteps, batch),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(
                jnp.abs(jnp.mean(rewards, axis=(0, 1)))
            )
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps + batch_size,
            )

            new_memory = MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "log_probs": jnp.zeros(num_envs),
                    "values": jnp.zeros(num_envs),
                },
            )

            return new_state, new_memory, metrics

        def make_initial_state(
            key: Any, hidden: jnp.ndarray
        ) -> Tuple[TrainingState, MemoryState]:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey = jax.random.split(key)

            if isinstance(obs_spec, dict):
                dummy_obs = {}
                for k, v in obs_spec.items():
                    dummy_obs[k] = jnp.zeros(shape=v)

            elif not tabular:
                dummy_obs = jnp.zeros(shape=obs_spec, dtype=float_precision)
                dummy_obs = dummy_obs.at[0].set(1)
                dummy_obs = dummy_obs.at[9].set(1)
                dummy_obs = dummy_obs.at[18].set(1)
                dummy_obs = dummy_obs.at[27].set(1)
            else:
                dummy_obs = jnp.zeros(shape=obs_spec)

            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            self.optimizer = optimizer
            return TrainingState(
                random_key=key,
                params=initial_params,
                opt_state=initial_opt_state,
                timesteps=0,
            ), MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            )

        def prepare_batch(
            traj_batch: NamedTuple, done: Any, action_extras: dict
        ):
            # Rollouts complete -> Training begins
            # Add an additional rollout step for advantage calculation
            _value = jax.lax.select(
                done,
                jnp.zeros_like(action_extras["values"]),
                action_extras["values"],
            )

            _value = jax.lax.expand_dims(_value, [0])
            # need to add final value here
            traj_batch = traj_batch._replace(
                behavior_values=jnp.concatenate(
                    [traj_batch.behavior_values, _value], axis=0
                )
            )
            return traj_batch

        # Initialise training state (parameters, optimiser state, extras).
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(random_key, jnp.zeros(1))
        self._prepare_batch = jax.jit(prepare_batch)
        self._sgd_step = jax.jit(sgd_step)

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
            "loss_entropy": 0,
            "entropy_cost": entropy_coeff_start,
        }

        # Initialize functions
        self._policy = policy
        self.player_id = player_id
        self.network = network

        # Other useful hyperparameters
        self._num_envs = num_envs  # number of environments
        self._num_minibatches = num_minibatches  # number of minibatches
        self._num_epochs = num_epochs  # number of epochs to use sample

    def reset_memory(self, memory, eval=False) -> MemoryState:
        num_envs = 1 if eval else self._num_envs
        memory = memory._replace(
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
        )
        return memory

    def update(
        self,
        traj_batch,
        obs: jnp.ndarray,
        state: TrainingState,
        mem: MemoryState,
    ):
        """Update the agent -> only called at the end of a trajectory"""
        _, _, mem = self._policy(state, obs, mem)

        traj_batch = self._prepare_batch(
            traj_batch, traj_batch.dones[-1, ...], mem.extras
        )
        state, mem, metrics = self._sgd_step(state, traj_batch)
        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        self._logger.metrics["loss_total"] = metrics["loss_total"]
        self._logger.metrics["loss_policy"] = metrics["loss_policy"]
        self._logger.metrics["loss_value"] = metrics["loss_value"]
        self._logger.metrics["loss_entropy"] = metrics["loss_entropy"]
        self._logger.metrics["entropy_cost"] = metrics["entropy_cost"]

        return state, mem, metrics


def make_agent(
    args,
    agent_args,
    obs_spec,
    action_spec,
    seed: int,
    num_iterations: int,
    player_id: int,
    tabular=False,
):
    """Make PPO agent"""
    print(f"Making network for {args.env_id}")
    if args.env_id == "coin_game":
        network = make_coingame_network(
            action_spec,
            tabular,
            agent_args.with_cnn,
            agent_args.separate,
            agent_args.hidden_size,
            agent_args.output_channels,
            agent_args.kernel_shape,
        )
    elif args.env_id == "InTheMatrix":
        network = make_ipditm_network(
            action_spec,
            agent_args.separate,
            agent_args.with_cnn,
            agent_args.hidden_size,
            agent_args.output_channels,
            agent_args.kernel_shape,
        )
    elif args.env_id in [
        "iterated_matrix_game",
        "iterated_tensor_game",
        "iterated_nplayer_tensor_game",
        "third_party_punishment",
        "third_party_random",
    ]:
        network = make_ipd_network(
            action_spec, tabular, agent_args.hidden_size
        )
    elif args.env_id == "Cournot":
        network = make_cournot_network(action_spec, agent_args.hidden_size)
    elif args.env_id == "Fishery":
        network = make_fishery_network(action_spec, agent_args.hidden_size)
    elif args.env_id == SarlRice.env_id:
        network = make_rice_sarl_network(action_spec, agent_args.hidden_size)
    elif args.env_id == Rice.env_id:
        network = make_rice_sarl_network(action_spec, agent_args.hidden_size)
    elif args.env_id == ClubRice.env_id:
        network = make_rice_sarl_network(action_spec, agent_args.hidden_size)
    elif args.runner == "sarl":
        network = make_sarl_network(action_spec)
    elif args.env_id in [
        IteratedMatrixGame.env_id,
        IteratedTensorGameNPlayer.env_id,
    ]:
        network = make_ipd_network(action_spec, True, agent_args.hidden_size)
    else:
        raise NotImplementedError(
            f"No ppo network implemented for env {args.env_id}"
        )

    # Optimizer
    transition_steps = (
        num_iterations * agent_args.num_epochs * agent_args.num_minibatches
    )

    if agent_args.lr_scheduling:
        scale = optax.inject_hyperparams(optax.scale)(step_size=-1.0)
        scheduler = optax.linear_schedule(
            init_value=agent_args.learning_rate,
            end_value=0,
            transition_steps=transition_steps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.max_gradient_norm),
            optax.scale_by_adam(eps=agent_args.adam_epsilon),
            optax.scale_by_schedule(scheduler),
            scale,
        )
        # optimizer = optax.inject_hyperparams(optimizer)(learning_rate=agent_args.learning_rate)

    else:
        scale = optax.inject_hyperparams(optax.scale)(step_size=-agent_args.learning_rate)
        optimizer = optax.chain(
            optax.clip_by_global_norm(agent_args.max_gradient_norm),
            optax.scale_by_adam(eps=agent_args.adam_epsilon),
            scale,
        )
        # optimizer = optax.inject_hyperparams(optimizer)(learning_rate=agent_args.learning_rate)

    # Random key
    random_key = jax.random.PRNGKey(seed=seed)

    agent = PPO(
        network=network,
        optimizer=optimizer,
        random_key=random_key,
        obs_spec=obs_spec,
        num_envs=args.num_envs,
        num_minibatches=agent_args.num_minibatches,
        num_epochs=agent_args.num_epochs,
        clip_value=agent_args.clip_value,
        value_coeff=agent_args.value_coeff,
        anneal_entropy=agent_args.anneal_entropy,
        entropy_coeff_start=agent_args.entropy_coeff_start,
        entropy_coeff_end=agent_args.entropy_coeff_end,
        entropy_coeff_horizon=agent_args.entropy_coeff_horizon,
        ppo_clipping_epsilon=agent_args.ppo_clipping_epsilon,
        gamma=agent_args.gamma,
        gae_lambda=agent_args.gae_lambda,
        tabular=tabular,
        player_id=player_id,
    )
    return agent


if __name__ == "__main__":
    pass
