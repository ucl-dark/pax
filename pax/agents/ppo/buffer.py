from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from dm_env import TimeStep


class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray


class TrajectoryBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent
    interacting with the environment. The buffer's capacity should equal
    the number of steps * number of environments + 1
    """

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_space: Tuple[int],  # (envs.observation_spec().num_values, )
        gru_dim: int = 1,
    ):

        # Buffer information
        self._num_envs = num_envs
        self._num_steps = (
            num_steps + 1
        )  # Take an additional rollout step for boostrapping value
        self._rollout_length = num_steps * num_envs

        # Environment specs
        self.obs_space = obs_space
        self.gru_dim = gru_dim

        # Initialise pointers
        self.ptr = 0

        # extra info
        self._num_added = 0
        self.full = False
        self.reset()

    def add(
        self,
        timestep: TimeStep,
        action: jnp.ndarray,
        log_prob: jnp.ndarray,
        value: jnp.ndarray,
        new_timestep: TimeStep,
        hidden: jnp.ndarray = None,
    ):
        """Append a batched time step to the buffer.
        Resets buffer and ptr if the buffer is full."""

        if self.full:
            self.reset()

        self.observations = self.observations.at[:, self.ptr].set(
            timestep.observation
        )
        self.actions = self.actions.at[:, self.ptr].set(action)
        self.behavior_log_probs = self.behavior_log_probs.at[:, self.ptr].set(
            log_prob
        )
        self.behavior_values = jax.lax.stop_gradient(
            self.behavior_values.at[:, self.ptr].set(value.flatten())
        )
        self.dones = self.dones.at[:, self.ptr].set(new_timestep.step_type)
        self.rewards = self.rewards.at[:, self.ptr].set(new_timestep.reward)

        if hidden is not None:
            self.hiddens = self.hiddens.at[:, self.ptr].set(hidden)

        self.ptr += 1
        self._num_added += self._num_envs

        if self.ptr == self._num_steps:
            self.full = True

    def sample(self):
        """Returns current data"""
        return Sample(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            behavior_log_probs=self.behavior_log_probs,
            behavior_values=self.behavior_values,
            dones=self.dones,
            hiddens=self.hiddens,
        )

    def size(self) -> int:
        return min(self._rollout_length, self._num_added)

    def fraction_filled(self) -> float:
        return self.size() / self._rollout_length

    def reset(self):
        """Resets the replay buffer. Called upon __init__ and when buffer is full"""
        self.ptr = 0
        self.full = False

        self.observations = jnp.zeros(
            (self._num_envs, self._num_steps, *self.obs_space)
        )

        self.actions = jnp.zeros(
            (
                self._num_envs,
                self._num_steps,
            ),
            dtype="int32",
        )

        self.behavior_log_probs = jnp.zeros(
            (
                self._num_envs,
                self._num_steps,
            )
        )
        self.rewards = jnp.zeros(
            (
                self._num_envs,
                self._num_steps,
            )
        )

        self.dones = jnp.zeros(
            (
                self._num_envs,
                self._num_steps,
            )
        )
        self.behavior_values = jnp.zeros(
            (
                self._num_envs,
                self._num_steps,
            )
        )
        self.hiddens = jnp.zeros(
            (self._num_envs, self._num_steps, self.gru_dim)
        )


if __name__ == "__main__":
    pass
