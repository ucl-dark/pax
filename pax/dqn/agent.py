# Adapted from https://github.com/deepmind/bsuite/blob/master/bsuite/baselines/jax/dqn/agent.py

from typing import Any, NamedTuple, Sequence

from bsuite.baselines import base
from bsuite.environments import catch
from dm_env import TimeStep
from pax.dqn.replay_buffer import Replay

import distrax
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax


class TrainingState(NamedTuple):
    """Holds the agent's training state."""

    params: hk.Params
    target_params: hk.Params
    opt_state: optax.GradientTransformation
    random_key: jnp.ndarray
    timesteps: int


class DQN(base.Agent):
    """A simple DQN agent using JAX."""

    def __init__(
        self,
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        network: NamedTuple,  # actually a Transform type
        optimizer: optax.GradientTransformation,
        random_key: Any,
        batch_size: int,
        epsilon: float,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
        player_id: int,
    ):
        @jax.jit
        def training_policy(
            params: hk.Params,
            observation: TimeStep,
            state: TrainingState,
            epsilon: float,
        ):
            """Agent training policy to select actions according to epsilon greedy"""
            key, subkey = jax.random.split(state.random_key)
            q_values = network.apply(
                params, observation
            )  # (num_envs, state space)
            e_greedy_dist = distrax.EpsilonGreedy(
                preferences=q_values, epsilon=epsilon
            )
            actions = e_greedy_dist.sample(seed=subkey)
            state = TrainingState(
                params=state.params,
                target_params=state.target_params,
                opt_state=state.opt_state,
                random_key=key,
                timesteps=state.timesteps,
            )
            return actions, state

        @jax.jit
        def evaluation_policy(
            params: hk.Params, observation: TimeStep, state: TrainingState
        ):
            """Agent evaluation policy to select actions according greedy"""
            key, subkey = jax.random.split(state.random_key)
            q_values = network.apply(params, observation)
            greedy_dist = distrax.Greedy(q_values)  # argument is preferences
            actions = greedy_dist.sample(seed=subkey)
            state = TrainingState(
                params=state.params,
                target_params=state.target_params,
                opt_state=state.opt_state,
                random_key=key,
                timesteps=state.timesteps,
            )
            return actions, state

        def loss(
            params: hk.Params,
            target_params: hk.Params,
            transitions: Sequence[jnp.ndarray],
        ) -> jnp.ndarray:
            """Computes the standard TD(0) Q-learning loss on batch of transitions."""
            # observation, action, reward, discount, observation
            o_tm1, a_tm1, r_t, d_t, o_t = transitions
            q_tm1 = network.apply(params, o_tm1)
            q_t = network.apply(target_params, o_t)
            batch_q_learning = jax.vmap(rlax.q_learning)
            td_error = batch_q_learning(q_tm1, a_tm1, r_t, discount * d_t, q_t)
            return jnp.mean(td_error**2)

        @jax.jit
        def sgd_step(
            state: TrainingState, transitions: Sequence[jnp.ndarray]
        ) -> TrainingState:
            """Performs an SGD step on a batch of transitions."""
            gradients = jax.grad(loss)(
                state.params, state.target_params, transitions
            )
            updates, new_opt_state = optimizer.update(
                gradients, state.opt_state
            )
            new_params = optax.apply_updates(state.params, updates)
            return TrainingState(
                params=new_params,
                target_params=state.target_params,
                opt_state=new_opt_state,
                random_key=state.random_key,
                timesteps=state.timesteps + 1,
            )

        def make_initial_state(key: Any, obs_spec) -> TrainingState:
            """Initialises the training state (parameters and optimiser state)."""
            key, subkey_1, subkey_2 = jax.random.split(key, 3)
            dummy_observation = np.zeros((1, obs_spec.num_values), jnp.float32)
            initial_params = network.init(subkey_1, dummy_observation)
            initial_target_params = network.init(subkey_2, dummy_observation)
            initial_opt_state = optimizer.init(initial_params)
            return TrainingState(
                params=initial_params,
                target_params=initial_target_params,
                opt_state=initial_opt_state,
                random_key=key,
                timesteps=0,
            )

        # Initialize the state
        self._state = make_initial_state(random_key, obs_spec)

        self._sgd_step = sgd_step
        self._training_policy = training_policy
        self._evaluation_policy = evaluation_policy
        self._replay = Replay(capacity=replay_capacity)

        # Store hyperparameters.
        self._num_actions = action_spec.num_values
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._epsilon = epsilon
        self._total_steps = 0
        self._min_replay_size = min_replay_size

        # Tracks the number of target updates
        self.target_step_updates = 0

        # Uniquely identifies the player as 1 or 2
        self.player_id = player_id

    def select_action(self, t: dm_env.TimeStep) -> jnp.array:
        """Selects batched actions based on epsilon greedy / greedy policy"""
        if self.eval:
            # Greedy policy, breaking ties uniformly at random.
            observation = t.observation[None, ...]
            action, self._state = self._evaluation_policy(
                self._state.target_params, observation, self._state
            )
            return action
        else:
            # Epsilon greedy policy, breaking ties uniformly at random
            action, self._state = self._training_policy(
                self._state.params, t.observation, self._state, self._epsilon
            )
            return action

    def update(
        self,
        timestep: dm_env.TimeStep,
        action: jnp.array,
        new_timestep: dm_env.TimeStep,
    ):

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
        transitions, key = self._replay.sample(
            self._batch_size, self._state.random_key
        )
        self._state = self._state._replace(random_key=key)
        self._state = self._sgd_step(self._state, transitions)

        # Periodically update target parameters.
        # Default update period is every 4 sgd steps.
        if self._state.timesteps % self._target_update_period == 0:
            self._state = self._state._replace(
                target_params=self._state.params
            )
            self.target_step_updates += 1


def default_agent(
    args,
    obs_spec: specs.Array,
    action_spec: specs.DiscreteArray,
    seed: int,
    player_id: int,
) -> base.Agent:
    """Initialize a DQN agent with default parameters."""
    random_key = jax.random.PRNGKey(seed=seed)

    def forward_fn(inputs: jnp.ndarray) -> jnp.ndarray:
        flat_inputs = hk.Flatten()(inputs)
        # Single layer network which acts as a lookup table
        mlp = hk.Linear(action_spec.num_values, with_bias=False)

        # Optimistic initializations (with constant weights)
        # mlp = hk.Linear(action_spec.num_values, w_init = hk.initializers.Constant(0.5), with_bias=False)

        # Larger MLP with a hidden layer of output size 10
        # mlp = hk.nets.MLP([10, action_spec.num_values], with_bias=False) #works alright with more layers
        action_values = mlp(flat_inputs)
        return action_values

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return DQN(
        obs_spec=obs_spec,
        action_spec=action_spec,
        network=network,
        optimizer=optax.adam(args.dqn.learning_rate),
        random_key=random_key,
        batch_size=args.dqn.batch_size,
        discount=args.dqn.discount,
        replay_capacity=args.dqn.replay_capacity,
        min_replay_size=args.dqn.min_replay_size,
        sgd_period=args.dqn.sgd_period,
        target_update_period=args.dqn.target_update_period,
        epsilon=args.dqn.epsilon,
        player_id=player_id,
    )
