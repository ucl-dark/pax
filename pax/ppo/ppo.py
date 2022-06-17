from multiprocessing.dummy import Array
from tkinter import N
from typing import Any, NamedTuple, Tuple, Dict
from pax.ppo.networks import make_network
from pax.ppo.buffer import TrajectoryBuffer
from pax import utils
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import jax
from dm_env import TimeStep


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


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: hk.Params
    opt_state: Any
    random_key: Any
    timesteps: int


class Logger:
    metrics: dict


class PPO:
    """A simple PPO agent using JAX"""

    def __init__(
        self,
        network: NamedTuple,  # Transform
        optimizer: optax.GradientTransformation,
        random_key: jnp.ndarray,
        obs_spec: Tuple,
        num_envs: int = 4,
        num_steps: int = 500,  # Total number of steps per batch
        num_minibatches: int = 16,
        num_epochs: int = 4,  # Number of times to use rollout data to train. ACME default = 1
        clip_value: bool = True,
        value_coeff: float = 0.5,
        anneal_entropy: bool = False,
        entropy_coeff_start: float = 0.1,
        entropy_coeff_end: float = 0.01,
        entropy_coeff_horizon: int = 3_000_000,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        @jax.jit
        def policy(
            params: hk.Params, observation: TimeStep, random_key: jnp.ndarray
        ):
            """Agent policy to select actions and calculate agent specific information"""
            key, subkey = jax.random.split(random_key)
            dist, values = network.apply(params, observation)
            actions = dist.sample(seed=subkey)
            log_prob, entropy = dist.log_prob(actions), dist.entropy()
            info = {
                "values": values,
                "log_probs": log_prob,
                "entropy": entropy,
                "random_key": key,
            }
            return actions, info

        def rollouts(
            buffer: TrajectoryBuffer,
            t: TimeStep,
            actions: np.array,
            t_prime: TimeStep,
            infos: dict,
        ) -> None:
            """Stores rollout in buffer"""
            buffer.add(
                t, actions, infos["log_probs"], infos["values"], t_prime
            )

        def gae_advantages(
            rewards: jnp.ndarray, values: jnp.ndarray, dones: jnp.ndarray
        ) -> jnp.ndarray:
            """Calculates the gae advantages from a sequence. Note that the
            arguments are of length = rollout length + 1"""
            # Only need up to the rollout length
            rewards = rewards[:-1]
            dones = dones[:-1]
            # 'Zero out' the terminated states
            discounts = gamma * (1 - dones)
            delta = rewards + discounts * values[1:] - values[:-1]
            advantage_t = [0.0]
            for t in reversed(range(delta.shape[0])):
                advantage_t.insert(
                    0, delta[t] + gae_lambda * discounts[t] * advantage_t[0]
                )
            advantages = jax.lax.stop_gradient(jnp.array(advantage_t[:-1]))
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

            # vmap
            batch_gae_advantages = jax.vmap(gae_advantages, in_axes=0)
            advantages, target_values = batch_gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_envs, num_steps, ..]
            (
                observations,
                actions,
                behavior_log_probs,
                behavior_values,
            ) = jax.tree_map(
                lambda x: x[:, :-1],
                (observations, actions, behavior_log_probs, behavior_values),
            )

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
            num_envs = target_values.shape[0]
            num_steps = target_values.shape[1]
            batch_size = num_envs * num_steps
            assert batch_size % num_minibatches == 0, (
                "Num minibatches must divide batch size. Got batch_size={}"
                " num_minibatches={}."
            ).format(batch_size, num_minibatches)

            batch = jax.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), trajectories
            )

            # Compute gradients.
            grad_fn = jax.grad(loss, has_aux=True)

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
                shuffled_batch = jax.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_map(
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

            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(
                jnp.abs(jnp.mean(rewards, axis=(0, 1)))
            )
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))

            new_state = TrainingState(
                params=params,
                opt_state=opt_state,
                random_key=key,
                timesteps=timesteps,
            )

            return new_state, metrics

        def make_initial_state(key: Any, obs_spec: Tuple) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(subkey, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                params=initial_params,
                opt_state=initial_opt_state,
                random_key=key,
                timesteps=0,
            )

        # Initialise training state (parameters and optimiser state).
        self._state = make_initial_state(random_key, obs_spec)

        # Stores information every time an action is taken such as log_prob, values, etc...
        self._infos = None

        # Initialize buffer and sgd
        self._trajectory_buffer = TrajectoryBuffer(
            num_envs, num_steps, obs_spec
        )
        self._sgd_step = sgd_step

        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {"total_steps": 0, "sgd_steps": 0}

        # Initialize functions
        self._policy = policy
        self._rollouts = rollouts

        # Other useful hyperparameters
        self._num_envs = num_envs  # number of environments
        self._num_steps = num_steps  # number of steps per environment
        self._batch_size = int(num_envs * num_steps)  # number in one batch
        self._num_minibatches = num_minibatches  # number of minibatches
        self._num_epochs = num_epochs  # number of epochs to use sample

    def select_action(self, t: TimeStep):
        """Selects action and updates info with PPO specific information"""
        actions, self._infos = self._policy(
            self._state.params, t.observation, self._state.random_key
        )
        self._state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._infos["random_key"],
            timesteps=self._state.timesteps,
        )
        return utils.to_numpy(actions)

    def update(
        self, t: TimeStep, actions: np.array, infos: dict, t_prime: TimeStep
    ):
        # Adds agent and environment info to buffer
        self._rollouts(
            buffer=self._trajectory_buffer,
            t=t,
            actions=actions,
            t_prime=t_prime,
            infos=infos,
        )

        # Log metrics
        self._total_steps += self._num_envs
        self._logger.metrics["total_steps"] += self._num_envs
        self._state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._infos["random_key"],
            timesteps=self._total_steps,
        )
        self._until_sgd += 1

        # Rollouts still in progress
        if self._until_sgd % (self._num_steps + 1) != 0:
            return

        # Rollouts complete -> Training begins
        sample = self._trajectory_buffer.sample()
        self._state, results = self._sgd_step(self._state, sample)
        self._logger.metrics["sgd_steps"] += (
            self._num_minibatches * self._num_epochs
        )
        self._logger.metrics["loss_total"] = results["loss_total"]
        self._logger.metrics["loss_policy"] = results["loss_policy"]
        self._logger.metrics["loss_value"] = results["loss_value"]
        self._logger.metrics["loss_entropy"] = results["loss_entropy"]


# TODO: action_spec, seed, and player_id not used in CartPole
def make_agent(args, obs_spec, action_spec, seed: int, player_id: int):
    """Make PPO agent"""
    # create a schedule for the optimizer
    batch_size = int(args.num_envs * args.num_steps)
    transition_steps = (
        args.total_timesteps
        / batch_size
        * args.num_epochs
        * args.num_minibatches
    )

    # TODO: make this more elegant
    # Make optimizer
    if args.scheduling:
        scheduler = optax.linear_schedule(
            init_value=args.learning_rate,
            end_value=0,
            transition_steps=transition_steps,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_gradient_norm),
            optax.scale_by_adam(eps=args.adam_epsilon),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1),
        )

    else:
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.max_gradient_norm),
            optax.scale_by_adam(eps=args.adam_epsilon),
            optax.scale(-args.learning_rate),
        )

    # network = hk.without_apply_rng(hk.transform(forward_fn))
    network = make_network(action_spec)

    return PPO(
        network=network,
        optimizer=optimizer,
        random_key=jax.random.PRNGKey(seed=args.seed),
        obs_spec=obs_spec,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        num_epochs=args.num_epochs,
        clip_value=args.clip_value,
        value_coeff=args.value_coeff,
        anneal_entropy=args.anneal_entropy,
        entropy_coeff_start=args.entropy_coeff_start,
        entropy_coeff_end=args.entropy_coeff_end,
        entropy_coeff_horizon=args.entropy_coeff_horizon,
        ppo_clipping_epsilon=args.ppo_clipping_epsilon,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )


if __name__ == "__main__":
    pass
