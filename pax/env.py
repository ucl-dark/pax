import enum
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from dm_env import Environment, TimeStep, specs, termination, transition

#              CC      CD     DC     DD
# payoffs are [(2, 2), (0,3), (3,0), (1, 1)]
# observations are one-hot representations


class State(enum.IntEnum):
    CC = 0
    CD = 1
    DC = 2
    DD = 3
    START = 4


_ACTIONS = (0, 1)  # Cooperate, Defect
_STATES = (0, 1, 2, 3, 4)  # CC, CD, DC, DD, START


class SequentialMatrixGame(Environment):
    def __init__(
        self, num_envs: int, payoff: list, episode_length: int
    ) -> None:
        self.payoff = jnp.array(payoff)
        self.episode_length = episode_length
        self.num_envs = num_envs
        self.n_agents = 2
        self._num_steps = 0
        self._reset_next_step = True

        # Dummy variable used to make Runner work with
        # regular and meta learning.
        self.num_trials = 1
        self.state = (0.0, 0.0)
        self.inner_episode_length = episode_length

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
        assert action_1.shape == (self.num_envs,)

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

    def runner_step(
        self,
        actions: Tuple[jnp.ndarray, jnp.ndarray],
        env_state: Tuple[float, float],
    ) -> Tuple[Tuple[TimeStep, TimeStep], Tuple[float, float]]:

        r1, r2 = self._get_reward(actions[0], actions[1])
        state = self._get_state(actions[0], actions[1])
        obs1, obs2 = self._observation(state)

        return (
            TimeStep(
                jnp.ones_like(r1, dtype=int),
                r1,
                jnp.zeros_like(r1, dtype=int),
                obs1,
            ),
            TimeStep(
                jnp.ones_like(r2, dtype=int),
                r2,
                jnp.zeros_like(r2, dtype=int),
                obs2,
            ),
        ), env_state

    def runner_reset(self, ndims):
        """Returns the first `TimeStep` of a new episode."""
        state = (0.0, 0.0)
        obs = obs = jax.nn.one_hot(State.START * jnp.ones(ndims), 5)
        discount = jnp.zeros(ndims, dtype=int)
        step_type = jnp.zeros(ndims, dtype=int)
        rewards = jnp.zeros(ndims)
        return (
            TimeStep(step_type, rewards, discount, obs),
            TimeStep(step_type, rewards, discount, obs),
        ), state

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        t_init, _ = self.runner_reset((self.num_envs))
        self._num_steps = 0
        return t_init

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, a1, a2) -> Tuple[jnp.array, jnp.array]:
        """Returns the rewards of a step"""
        # Example payoffs
        #             CC      CD     DC     DD
        # IPD       = [[2,2], [0,3], [3,0], [1,1]]
        # Stag hunt = [[4,4], [1,3], [3,1], [2,2]]
        # BotS      = [[3,2], [0,0], [0,0], [2,3]]
        # Chicken   = [[0,0], [-1,1],[1,-1],[-2,-2]]
        cc_p1 = self.payoff[0][0] * (a1 - 1.0) * (a2 - 1.0)
        cc_p2 = self.payoff[0][1] * (a1 - 1.0) * (a2 - 1.0)

        cd_p1 = self.payoff[1][0] * (1.0 - a1) * a2
        cd_p2 = self.payoff[1][1] * (1.0 - a1) * a2

        dc_p1 = self.payoff[2][0] * a1 * (1.0 - a2)
        dc_p2 = self.payoff[2][1] * a1 * (1.0 - a2)

        dd_p1 = self.payoff[3][0] * a1 * a2
        dd_p2 = self.payoff[3][1] * a1 * a2

        r1 = cc_p1 + dc_p1 + cd_p1 + dd_p1
        r2 = cc_p2 + dc_p2 + cd_p2 + dd_p2

        return r1, r2

    @partial(jax.jit, static_argnums=(0,))
    def _get_state(self, a1: jnp.array, a2: jnp.array) -> jnp.array:
        return (
            0 * (1 - a1) * (1 - a2)
            + 1 * (1 - a1) * a2
            + 2 * a1 * (1 - a2)
            + 3 * (a1) * (a2)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _observation(self, state: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # each agent wants to assume they are agent 1 canonicaly:
        # observation also changes environment to be state space
        o_state = jnp.array(state, copy=True)

        # switch the assymetric states (CD and DC)
        # 0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3
        s0 = state
        s1 = State.CD - state
        s2 = State.DC - state
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

        return jax.nn.one_hot(o_state, len(State)), jax.nn.one_hot(
            state, len(State)
        )


class IteratedPrisonersDilemma(SequentialMatrixGame):
    """Define iterated prisoner's dilemma"""

    def __init__(self, episode_length: int, num_envs: int) -> None:
        # (CC, DC, CD, DD)
        self.payoff = jnp.array([[2, 2], [0, 3], [3, 0], [1, 1]])
        super().__init__(episode_length, num_envs, self.payoff)


class StagHunt(SequentialMatrixGame):
    """Define stag hunt"""

    def __init__(self, episode_length: int, num_envs: int) -> None:
        # (CC, DC, CD, DD)
        self.payoff = jnp.array([[4, 4], [1, 3], [3, 1], [2, 2]])
        super().__init__(episode_length, num_envs, self.payoff)


class BattleOfTheSexes(SequentialMatrixGame):
    """Define Battle of the Sexes"""

    def __init__(self, episode_length: int, num_envs: int) -> None:
        # (CC, DC, CD, DD)
        self.payoff = jnp.array([[3, 2], [0, 0], [0, 0], [2, 3]])
        super().__init__(episode_length, num_envs, self.payoff)


class Chicken(SequentialMatrixGame):
    """Define Chicken"""

    def __init__(self, episode_length: int, num_envs: int) -> None:
        # (CC, DC, CD, DD)
        self.payoff = jnp.array([[0, 0], [-1, 1], [1, -1], [-2, -2]])
        super().__init__(episode_length, num_envs, self.payoff)


if __name__ == "__main__":
    env = SequentialMatrixGame(
        5, 1, jnp.array([[2, 2], [0, 3], [3, 0], [1, 1]])
    )
