from typing import Dict, Any, Callable, NamedTuple, Sequence, Tuple
import hydra
import jax.numpy as jnp
from pax.ppo.networks import (
    critic_network,
    actor_network,
    CategoricalValueHead,
)
from pax.ppo.buffer import TrajectoryBuffer
import haiku as hk
import optax
import jax
import distrax
import rlax

# import wandb
# import acme
import dm_env

# import os


class TrainingState(NamedTuple):
    """Training state consists of network parameters and optimiser state."""

    params: hk.Params
    # actor_params: hk.Params
    opt_state: Any
    random_key: Any


# TODO: Complete PPO implementation
class PPO:
    """A simple PPO agent using JAX"""

    # TODO: complete init function
    def __init__(
        self,
        obs_spec,
        action_spec,
        optimizer: optax.GradientTransformation,
        rng: hk.PRNGSequence,
        player_id: int,
        args: dict,
    ):
        # init random keys
        key = jax.random.PRNGKey(args.seed)
        key, key_state = jax.random.split(key)

        # define and init network
        def forward_fn(inputs):
            layers = []
            layers.extend(
                [
                    hk.nets.MLP([2, 2], activate_final=True),
                    CategoricalValueHead(
                        num_values=action_spec
                    ),  # TODO: if doesn't work, replace with 2
                ]
            )
            policy_value_network = hk.Sequential(layers)
            return policy_value_network(inputs)

        network = hk.without_apply_rng(hk.transform(forward_fn))
        # TODO: Remove below line when replacing get_action_and_value.
        self.network = network

        # initialize state with network and optimizer params
        dummy_obs = jnp.zeros((obs_spec))
        dummy_obs = jax.tree_map(
            lambda x: jnp.expand_dims(x, axis=0), dummy_obs
        )  # Dummy 'sequence' dim.
        params = network.init(next(rng), dummy_obs)
        opt_state = optimizer.init(params)
        self._state = TrainingState(
            params=params, opt_state=opt_state, random_key=key_state
        )
        # TODO: init buffer
        # self.buffer= TrajectoryBuffer()

        def gae_advantages(
            rewards: jnp.array, discounts: jnp.array, values: jnp.array
        ) -> Tuple[jnp.ndarray, jnp.array]:
            """Uses truncated GAE to compute advantages."""
            # Apply reward clipping.
            rewards = jnp.clip(
                rewards, -args.max_abs_reward, args.max_abs_reward
            )

            advantages = rlax.truncated_generalized_advantage_estimation(
                rewards[:-1], discounts[:-1], args.gae_lambda, values
            )
            advantages = jax.lax.stop_gradient(advantages)

            # Exclude the bootstrap value
            target_values = values[:-1] + advantages
            target_values = jax.lax.stop_gradient(target_values)

            return advantages, target_values

        # TODO: complete loss function
        def loss(
            params: dict,
            observations,  # ?
            actions: jnp.array,
            behavior_log_probs: jnp.array,
            target_values: jnp.array,
            advantages: jnp.array,
            behavior_values: jnp.array,
        ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:  # ?
            pass
            """Surrogate loss using clipped probability ratios."""

            # # get distribution and parameters
            # distribution, values = self.network.apply(self._state.params, observations)

            # # Get log probabilities and entropy
            # log_prob = distribution.log_prob(actions)
            # entropy = distribution.entropy()

            # # Compute importance sampling weights: current policy / behavior policy.
            # rhos = jnp.exp(log_probs - behaviour_log_probs)

            # policy_loss = rlax.clipped_surrogate_pg_loss(rhos, advantages, ppo_clipping_epsilon)

            # # TODO: Understand this
            # # Value function loss. Exclude the bootstrap value
            # unclipped_value_error = target_values - values
            # unclipped_value_loss = unclipped_value_error ** 2

            # # Value loss:
            # value_loss = jnp.mean(unclipped_value_loss)

            # # Entropy loss: entropy regularizer
            # entropy_loss = -jnp.mean(entropy)

            # # TODO: value_cost = 1. in the ACME implementation?
            # value_cost = 1
            # # TODO: entropy_cost = 0 in the ACME implementation?
            # entropy_cost = 0
            # # this is the sum of the loss from the paper
            # total_loss = (policy_loss+value_loss * value_cost + entropy_loss * entropy_cost)
            # return total_loss, {
            #     'loss_total': total_loss,
            #     'loss_policy': policy_loss,
            #     'loss_value': value_loss,
            #     'loss_entropy': entropy_loss
            # }

        # TODO: complete sgd step
        # @jax.jit
        def sgd_step(state: TrainingState, transitions) -> TrainingState:
            """Performs a minibatch SGD step, returning new state."""
            # observations, actions, rewards, termination, extra = (5 things)
            #
            # TODO: mimic this layout
            # """Performs an SGD step on a batch of transitions."""
            # gradients = jax.grad(loss)(
            #     state.params, state.target_params, transitions
            # )
            # updates, new_opt_state = optimizer.update(
            #     gradients, state.opt_state
            # )
            # new_params = optax.apply_updates(state.params, updates)

            # grads = jax.grad(loss)(params)(buffer_output)
            # new_state = TrainingState(
            #     params=new_params,
            #     opt_state=new_opt_state,
            #     random_key=key)
            pass
            # return new_state

        # self._forward = jax.jit(critic_network.apply)
        # self._critic_forward = critic_network.apply
        # self._actor_forward = actor_network.apply

    # TODO: complete select_action()
    def select_action(self, obs):
        # distribution_params = self._actor_forward(
        #     self._state.actor_params, obs
        # )  # aka logits
        # actions = jax.random.categorical(key, logits=distribution_params)
        pass
        # return actions

    # TODO: This will eventually not be here.
    # def get_action_and_value(self, key, obs, action=None):
    #     """Return the logits, action"""
    #     distribution_params = self._actor_forward(self._state.actor_params, obs) # aka logits
    #     values = self._critic_forward(self._state.critic_params, obs)
    #     dist = distrax.Categorical(distribution_params)
    #     if action is None:
    #         actions = jax.random.categorical(key, logits=distribution_params)
    #     log_prob = dist.log_prob(actions)
    #     entropy = dist.entropy()
    #     return actions, log_prob, entropy, values

    # TODO: remove this and replcae with the forward pass
    def get_action_and_value(self, key, obs, action=None):
        dist, values = self.network.apply(self._state.params, obs)
        actions = dist.sample(seed=key)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_prob, entropy, values

    # TODO: complete update()
    def update(
        self,
        timestep: dm_env.TimeStep,
        action: jnp.array,
        new_timestep: dm_env.TimeStep,
        key,
    ):

        # TODO: Add batch to storage variables
        # self.buffer.add(state, action, reward, discount, new_state, log_prob, value, done)
        # TODO: Perform SGD for each minibatch of the batch
        self._replay.add_batch(
            [
                timestep.observation,
                action,
                new_timestep.reward,
                new_timestep.discount,
                new_timestep.observation,
            ],
            batch_size=len(timestep.observation),  # num envs
        )

        # Increment number of training steps taken for plotting purposes
        self._total_steps += 1
        if self._total_steps % self._sgd_period != 0:
            return

        if self._replay.size < self._min_replay_size:
            return

        # Do a batch of SGD.
        transitions = self._replay.sample(self._batch_size, key)
        self._state = self._sgd_step(self._state, transitions)

        # Periodically update target parameters.
        # Default update period is every 4 sgd steps.
        if self._state.step % self._target_update_period == 0:
            self._state = self._state._replace(
                target_params=self._state.params
            )
            self.target_step_updates += 1


def make_agent(args, obs_spec, action_spec, seed: int, player_id: int):
    """Make PPO agent"""
    # create a schedule for the optimizer
    scheduler = optax.linear_schedule(
        init_value=args.learning_rate,
        end_value=0,
        transition_steps=args.total_timesteps,
    )
    optimizer = optax.adam(learning_rate=scheduler, eps=args.eps)

    return PPO(
        obs_spec=obs_spec,
        action_spec=action_spec,
        optimizer=optimizer,
        rng=hk.PRNGSequence(seed),
        player_id=player_id,
        args=args,
    )


if __name__ == "__main__":
    pass
