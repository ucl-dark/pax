from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
from dm_env import TimeStep
import numpy as np


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray  # (t+1)
    actions: jnp.ndarray  # (t+1)
    rewards: jnp.ndarray  # (t+1)
    behavior_log_probs: jnp.ndarray  # (t+1)
    behavior_values: jnp.ndarray  # (t+1)
    dones: jnp.ndarray  # (t+1)


class TrajectoryBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent
    interacting with the environment. The buffer's capacity should equal
    the number of steps * number of environments + 1
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_space: Tuple[int],  # (envs.observation_spec().num_values, )
    ):

        # Buffer information
        self.num_steps = (
            num_steps + 1
        )  # Take an additional rollout step for boostrapping value
        self.num_envs = num_envs
        self._rollout_length = self.num_steps * num_envs

        # Environment and agent specs
        self.obs_space = obs_space

        # Initialise pointers
        self.ptr = 0

        # additional PPO specific
        self.value = None
        self.log_prob = None
        self.returns = None
        self.advantage = None

        # extra info
        self._num_added = 0
        self.full = False
        self.reset()

    # TODO: implement add()
    def add(
        self,
        timestep: TimeStep,
        action: jnp.ndarray,
        log_prob: jnp.ndarray,
        value: jnp.ndarray,
        new_timestep: TimeStep,
    ):
        """Append a batched time step to the buffer.
        Resets buffer and ptr if the buffer is full."""

        # Logic for larger buffer.
        # update variables
        # q, _ = divmod(self.ptr, self._capacity)

        # if q:
        #     self.ptr = 0

        if self.full:
            self.reset()

        self.observations = self.observations.at[self.ptr].set(
            timestep.observation
        )
        self.actions = self.actions.at[self.ptr].set(action)
        self.behavior_log_probs = self.behavior_log_probs.at[self.ptr].set(
            log_prob
        )
        self.behavior_values = jax.lax.stop_gradient(
            self.behavior_values.at[self.ptr].set(value.flatten())
        )
        self.dones = self.dones.at[self.ptr].set(timestep.step_type)
        self.rewards = self.rewards.at[self.ptr].set(new_timestep.reward)

        self.ptr += 1
        self._num_added += self.num_envs

        if self.ptr == self.num_steps:
            self.full = True

    def sample(self):
        """Returns current data"""
        # start = self.ptr - num_steps - 1 # We sample one extra step for bootstrapping
        # return Sample(observations=self.observations[start:self.ptr],
        #               actions=self.actions[start:self.ptr],
        #               rewards=self.rewards[start:self.ptr],
        #               behavior_log_probs=self.behavior_log_probs[start:self.ptr],
        #               behavior_values=self.behavior_values[start:self.ptr],
        #               dones=self.dones[start:self.ptr]
        #               )

        return Sample(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            behavior_log_probs=self.behavior_log_probs,
            behavior_values=self.behavior_values,
            dones=self.dones,
        )

    def size(self) -> int:
        return min(self._capacity, self._num_added)

    def fraction_filled(self) -> float:
        return self.size / self._capacity

    # TODO: implement reset()
    def reset(self):
        """ "Resets the replay buffer
        Called after the buffer reaches size=num_envs*step_size"""
        # TODO: add types
        self.ptr = 0
        self.full = False

        # TODO: backwards compatability

        self.observations = jnp.zeros(
            (self.num_steps, self.num_envs, *self.obs_space)
        )

        self.actions = jnp.zeros(
            (
                self.num_steps,
                self.num_envs,
            ),
            dtype="int32",
        )

        self.behavior_log_probs = jnp.zeros(
            (
                self.num_steps,
                self.num_envs,
            )
        )
        self.rewards = jnp.zeros(
            (
                self.num_steps,
                self.num_envs,
            )
        )

        self.dones = jnp.zeros(
            (
                self.num_steps,
                self.num_envs,
            )
        )
        self.behavior_values = jnp.zeros(
            (
                self.num_steps,
                self.num_envs,
            )
        )


def test_reset():
    capacity = 20
    obs_space = (4,)
    action_space = 2
    buffer = TrajectoryBuffer(capacity, obs_space, action_space)
    buffer.reset()
    print("States", buffer.states.shape)
    assert buffer.states.shape == (capacity, *obs_space)
    print("Actions", buffer.actions.shape)
    assert buffer.actions.shape == (capacity,)
    print("Log probabilities", buffer.logprobs.shape)
    assert buffer.logprobs.shape == (capacity,)
    # print(buffer.rewards)
    # print(buffer.dones)
    # print(buffer.values)


def test_add():
    capacity = 12
    obs_space = (4,)
    action_space = 2

    # Create and reset buffer
    buffer = TrajectoryBuffer(capacity, obs_space, action_space)
    buffer.reset()

    # Create and add dummy data to buffer
    sample_batch_size = 4
    state = jnp.ones(shape=(4, 4))
    action = jnp.ones(shape=(4,))
    buffer.add(state, action, sample_batch_size)
    np.testing.assert_array_equal(buffer.states[0:4], state)
    np.testing.assert_array_equal(buffer.actions[0:4], action)

    state = 2 * jnp.ones(shape=(4, 4))
    action = jnp.ones(shape=(4,))
    buffer.add(state, action, sample_batch_size)
    np.testing.assert_array_equal(buffer.states[4:8], state)
    np.testing.assert_array_equal(buffer.actions[4:8], action)

    # state = 2 * jnp.ones(shape=(4, 4))
    # buffer.add(state, 4)
    # np.testing.assert_array_equal(buffer.states[4:8], state)
    # state = 3 * jnp.ones(shape=(4, 4))
    # buffer.add(state, 4)
    # np.testing.assert_array_equal(buffer.states[8:12], state)

    # # Test buffer reset if adding another sequence past capacity
    # state = 4 * jnp.ones(shape=(4, 4))
    # buffer.add(state, 4)
    # np.testing.assert_array_equal(buffer.states[0:4], state)


def test_sample():
    capacity = 12
    obs_space = (4,)
    action_space = 2

    # Create and reset buffer
    buffer = TrajectoryBuffer(capacity, obs_space, action_space)
    buffer.reset()

    # Create and add dummy data to buffer
    sample_batch_size = 4
    expected_state1 = 1 * jnp.ones(shape=(sample_batch_size, 4))
    expected_state2 = 2 * jnp.ones(shape=(sample_batch_size, 4))
    expected_state3 = 3 * jnp.ones(shape=(sample_batch_size, 4))
    expected_action = jnp.ones(shape=(sample_batch_size,))
    buffer.add(expected_state1, expected_action, sample_batch_size)
    buffer.add(expected_state2, expected_action, sample_batch_size)
    buffer.add(expected_state3, expected_action, sample_batch_size)

    # Test generator
    key = jax.random.PRNGKey(0)  # set dummy key
    sample_generator = buffer.sample(key, num_minibatches=3)
    transition = next(sample_generator)
    state1, action = transition[0], transition[1]
    np.testing.assert_array_equal(expected_state1, state1)
    np.testing.assert_array_equal(expected_action, action)
    # np.testing.assert_array_equal((state2, action), next(sample_generator))
    # np.testing.assert_array_equal((state3, action), next(sample_generator))


if __name__ == "__main__":
    test_reset()
    test_add()
    test_sample()
