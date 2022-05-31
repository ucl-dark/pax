import enum
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from dm_env import (
    Environment,
    TimeStep,
    restart,
    specs,
    termination,
    transition,
)

# payoffs are [(2, 2), (3,0), (0,3), (1, 1)]
# observations are one-hot representations

class State(enum.IntEnum):
    CC = 0
    DC = 1
    CD = 2
    DD = 3
    START = 4


_ACTIONS = (0, 1)  # Cooperate, Defect
_STATES = (0, 1, 2, 3, 4)  # CC, DC, CD, DD, START


class IteratedPrisonersDilemma(Environment):
    def __init__(self, episode_length: int, num_envs: int) -> None:
        self.episode_length = episode_length
        self.num_envs = num_envs
        self.n_agents = 2

        self._num_steps = 0
        self._reset_next_step = True

    def step(
        self,
        actions: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[TimeStep, TimeStep]:
        """
        takes a tuple of batched actions ([1,2], [2,1]) and takes a step in the environment
        """
        if self._reset_next_step:
            return self.reset()
        action_1, action_2 = actions
        self._num_steps += 1
        assert action_1.shape == action_2.shape
        assert action_1.shape == (self.num_envs, 1)

        r_1, r_2 = self._get_reward(action_1, action_2)
        state = self._get_state(action_1, action_2)
        obs_1, obs_2 = self._observation(state)

        if self._num_steps == self.episode_length:
            self._reset_next_step = True
            return termination(reward=r_1, observation=obs_1), termination(
                reward=r_2, observation=obs_2
            )

        return transition(reward=r_1, observation=obs_1), transition(
            reward=r_2, observation=obs_2
        )

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(num_values=len(State), name="previous turn")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=len(_ACTIONS), name="action"
        )

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        obs_1, obs_2 = self._observation(
            State.START * jnp.ones((self.num_envs, 1))
        )
        self._num_steps = 0
        return restart(obs_1), restart(obs_2)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, a1, a2) -> Tuple[jnp.array, jnp.array]:
        cc = 2.0 * (a1 - 1.0) * (a2 - 1.0)
        dd = a1 * a2
        dc = 3.0 * a1 * (1.0 - a2)
        cd = 3.0 * (1.0 - a1) * a2
        r1 = cc + dd + dc
        r2 = cc + dd + cd
        return r1, r2

    @partial(jax.jit, static_argnums=(0,))
    def _get_state(self, a1: jnp.array, a2: jnp.array) -> jnp.array:
        return (
            0 * (1 - a1) * (1 - a2)
            + 1 * (a1) * (1 - a2)
            + 2 * (1 - a1) * (a2)
            + 3 * (a1) * (a2)
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def _observation(self, state: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # each agent wants to assume they are agent 1 canonicaly:
        # observation also changes environment to be state space
        o_state = jnp.array(state, copy=True)

        # switch the assymetric states (CD and DC)
        # 0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3
        s0 = state
        s1 = State.DC - state
        s2 = State.CD - state
        s3 = State.DD - state
        s4 = State.START - state

        # coefficients are calculated to guarantee switch we want
        state = (
            0 * s1 * s2 * s3 * s4
            + (2 / (1 * 2 * 3)) * s0 * s2 * s3 * s4
            + (1 / (2 * (-1) * 1 * 2)) * s0 * s1 * s3 * s4
            + (3 / (3 * (-2) * (-1) * 1)) * s0 * s1 * s2 * s4
            + (4 / (4 * (-3) * (-2) * (-1))) * s0 * s1 * s2 * s3
        )

        # make this one hot
        return jax.nn.one_hot(o_state, len(State)).squeeze(1), jax.nn.one_hot(
            state, len(State)
        ).squeeze(1)