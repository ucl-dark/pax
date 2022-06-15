from multiprocessing.dummy import Array
from typing import Any, NamedTuple, Tuple, Dict
from pax.ppo.networks import CategoricalValueHead, forward_fn
from pax.ppo.buffer import TrajectoryBuffer
from pax import utils
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import jax
import distrax
import rlax
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
        buffer_capacity: int,
        num_envs: int,
        num_steps: int,  # Total number of steps per batch
        num_minibatches: int,
        entropy_cost: float,
        value_cost: float,
        ppo_clipping_epsilon: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_epochs: int = 4,  # Number of times to use rollout data to train. ACME default = 1
    ):
        @jax.jit
        def policy(
            params: hk.Params, observation: TimeStep, random_key: jnp.ndarray
        ):
            """Agent policy to select actions and calculate agent specific information"""
            key = random_key
            key1, key2 = jax.random.split(key)
            dist, values = network.apply(params, observation)
            actions = dist.sample(seed=key1)
            log_prob, entropy = dist.log_prob(actions), dist.entropy()
            info = {
                "values": values,
                "log_probs": log_prob,
                "entropy": entropy,
                "random_key": key2,
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
            """Calculates the gae advantages"""
            # Only need bootstrapped value
            rewards = rewards[:-1]
            dones = dones[:-1]
            discounts = gamma * (
                1 - dones
            )  # product of these gives which episodes are terminated states
            # advantages = rlax.truncated_generalized_advantage_estimation(
            #     rewards[:-1],
            #     dones[:-1],
            #     gae_lambda,
            #     values)
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

            # Pure JAX implementation
            # policy_loss1 = -advantages * ratio
            # policy_loss2 = -advantages * jax.lax.clamp(min=1-args.clip_coef, x=ratio, max=1+args.clip_coef)
            # policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

            # RLAX implementation
            # policy_loss = rlax.clipped_surrogate_pg_loss(rhos, advantages, ppo_clipping_epsilon)

            # Policy loss
            clipped_ratios_t = jnp.clip(
                rhos, 1.0 - ppo_clipping_epsilon, 1.0 + ppo_clipping_epsilon
            )
            clipped_objective = jnp.fmin(
                rhos * advantages, clipped_ratios_t * advantages
            )
            policy_loss = -jnp.mean(clipped_objective)

            # Value function loss
            unclipped_value_error = target_values - values
            unclipped_value_loss = unclipped_value_error**2
            value_loss = jnp.mean(unclipped_value_loss)

            # Entropy loss: entropy regularizer
            entropy_loss = -jnp.mean(entropy)

            # Total loss: Minimize policy loss and value loss; maximize entropy
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

            # non-vmapped version:
            advantages, target_values = gae_advantages(
                rewards=rewards, values=behavior_values, dones=dones
            )

            # VMAP
            # batch_gae_advantages = jax.vmap(gae_advantages, in_axes=0)
            # advantages, target_values = batch_gae_advantages(
            #     rewards=rewards,
            #     values=behavior_values,
            #     dones=dones
            #     )

            # Exclude the last step - it was only used for bootstrapping.
            # The shape is [num_steps, num_envs, ..]
            (
                observations,
                actions,
                behavior_log_probs,
                behavior_values,
            ) = jax.tree_map(
                lambda x: x[:-1, :],
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

            # Concatenate all trajectories. Reshape from [num_steps, num_envs,..]
            # to [num_steps * num_sequences,..]
            assert len(target_values.shape) > 1
            num_steps = target_values.shape[0]
            num_sequences = target_values.shape[1]
            batch_size = num_sequences * num_steps
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
                carry: Tuple[hk.Params, optax.OptState],
                minibatch: Batch,
            ) -> Tuple[
                Tuple[hk.Params, optax.OptState], Dict[str, jnp.ndarray]
            ]:
                """Performs model update for a single minibatch."""
                params, opt_state = carry
                # Normalize advantages at the minibatch level before using them.
                advantages = (
                    minibatch.advantages
                    - jnp.mean(minibatch.advantages, axis=0)
                ) / (jnp.std(minibatch.advantages, axis=0) + 1e-8)
                gradients, metrics = grad_fn(
                    params,
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
                return (params, opt_state), metrics

            def model_update_epoch(
                carry: Tuple[jnp.ndarray, hk.Params, optax.OptState, Batch],
                unused_t: Tuple[()],
            ) -> Tuple[
                Tuple[jnp.ndarray, hk.Params, optax.OptState, Batch],
                Dict[str, jnp.ndarray],
            ]:
                """Performs model updates based on one epoch of data."""
                key, params, opt_state, batch = carry
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

                (params, opt_state), metrics = jax.lax.scan(
                    model_update_minibatch,
                    (params, opt_state),
                    minibatches,
                    length=num_minibatches,
                )
                return (key, params, opt_state, batch), metrics

            params = state.params
            opt_state = state.opt_state

            # Repeat training for the given number of epoch, taking a random
            # permutation for every epoch.
            # signature is scan(function, carry, tuple to iterate over, length)
            (key, params, opt_state, _), metrics = jax.lax.scan(
                model_update_epoch,
                (state.random_key, params, opt_state, batch),
                (),
                length=num_epochs,
            )

            metrics = jax.tree_map(jnp.mean, metrics)
            metrics["rewards_mean"] = jnp.mean(
                jnp.abs(jnp.mean(rewards, axis=(0, 1)))
            )
            metrics["rewards_std"] = jnp.std(rewards, axis=(0, 1))

            new_state = TrainingState(
                params=params, opt_state=opt_state, random_key=key
            )

            return new_state, metrics

        def make_initial_state(key: Any, obs_spec: Tuple) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key_init, key_state = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=obs_spec)
            dummy_obs = utils.add_batch_dim(dummy_obs)
            initial_params = network.init(key_init, dummy_obs)
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                params=initial_params,
                opt_state=initial_opt_state,
                random_key=key_state,
            )

        # Initialise training state (parameters and optimiser state).
        self._state = make_initial_state(random_key, obs_spec)

        # Stores information every time an action is taken such as log_prob, values, etc...
        self._infos = None

        # Initialize buffer and sgd
        self._trajectory_buffer = TrajectoryBuffer(
            num_steps, num_envs, obs_spec
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
        self._num_minibatches = num_minibatches
        self._batch_size = int(num_envs * num_steps)
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._num_epochs = num_epochs

    def select_action(self, t: TimeStep):
        """Selects action and updates info with PPO specific information"""
        actions, self._infos = self._policy(
            self._state.params, t.observation, self._state.random_key
        )
        self._state = TrainingState(
            params=self._state.params,
            opt_state=self._state.opt_state,
            random_key=self._infos["random_key"],
        )
        return utils.to_numpy(actions)

    def update(
        self, t: TimeStep, actions: np.array, infos: dict, t_prime: TimeStep
    ):

        # Update global count of steps taken
        self._total_steps += self._num_envs
        self._until_sgd += 1
        self._logger.metrics["total_steps"] += 1 * self._num_envs

        # Adds agent and environment info to buffer
        self._rollouts(
            buffer=self._trajectory_buffer,
            t=t,
            actions=actions,
            t_prime=t_prime,
            infos=infos,
        )

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


def make_agent(args, obs_spec, action_spec, seed: int, player_id: int):
    """Make PPO agent"""
    # create a schedule for the optimizer
    batch_size = int(args.num_envs * args.num_steps)
    transition_steps = args.total_timesteps / batch_size * 4

    # Scheduled learning rate
    scheduler = optax.linear_schedule(
        init_value=args.learning_rate,
        end_value=0,
        transition_steps=transition_steps,
    )

    # Fixed learning rate
    # scheduler = args.learning_rate
    optimizer = optax.adam(learning_rate=scheduler, eps=args.eps)
    network = hk.without_apply_rng(hk.transform(forward_fn))

    return PPO(
        network=network,  # Transform
        optimizer=optimizer,
        random_key=jax.random.PRNGKey(seed=args.seed),
        obs_spec=obs_spec,
        buffer_capacity=args.buffer_capacity,
        num_envs=args.num_envs,
        num_steps=args.num_steps,  # Total number of steps per batch
        num_minibatches=args.num_minibatches,
        entropy_cost=args.ent_coef,
        value_cost=args.vf_coef,
        ppo_clipping_epsilon=args.clip_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_epochs=args.update_epochs,
    )


if __name__ == "__main__":
    pass
