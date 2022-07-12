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

#              CC      DC     CD     DD
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


class InfiniteMatrixGame(Environment):
    def __init__(
        self, num_envs: int, payoff: list, episode_length: int, gamma: float
    ) -> None:
        self.payoff = jnp.array([payoff[0], payoff[2], payoff[1], payoff[3]])
        payout_mat_1 = jnp.array([[r[0] for r in self.payoff]])
        payout_mat_2 = jnp.array([[r[1] for r in self.payoff]])

        def _step(theta1, theta2):
            p_1_0 = theta1[4:5]
            p_2_0 = theta2[4:5]
            p = jnp.concatenate(
                [
                    p_1_0 * p_2_0,
                    p_1_0 * (1 - p_2_0),
                    (1 - p_1_0) * p_2_0,
                    (1 - p_1_0) * (1 - p_2_0),
                ]
            )
            p_1 = jnp.reshape(theta1[0:4], (4, 1))
            p_2 = jnp.reshape(theta2[0:4], (4, 1))
            P = jnp.concatenate(
                [
                    p_1 * p_2,
                    p_1 * (1 - p_2),
                    (1 - p_1) * p_2,
                    (1 - p_1) * (1 - p_2),
                ],
                axis=1,
            )
            M = jnp.matmul(p, jnp.linalg.inv(jnp.eye(4) - gamma * P))
            L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (4, 1)))
            L_2 = jnp.matmul(M, jnp.reshape(payout_mat_2, (4, 1)))
            return L_1 * (1 - gamma), L_2 * (1 - gamma)

        self._step_2 = jax.vmap(_step)
        self._num_steps = 0
        self._reset_next_step = True
        self.num_envs = num_envs
        self.n_agents = 2
        self.episode_length = episode_length

    def step(
        self, actions: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> Tuple[TimeStep, TimeStep]:
        """
        takes a tuple of batched policies and produce value functions from infinite game
        policy of form [B, 5]
        """
        if self._reset_next_step:
            return self.reset()

        action_1, action_2 = actions
        self._num_steps += 1
        assert action_1.shape == action_2.shape
        assert action_1.shape == (self.num_envs, 5)

        r1, r2, obs1, obs2 = self._step(
            action_1,
            action_2,
        )
        # unwrap rewards for buffer
        r1 = r1[:, 0]
        r2 = r2[:, 0]
        if self._num_steps == self.episode_length:
            self._reset_next_step = True
            return termination(reward=r1, observation=obs1), termination(
                reward=r2, observation=obs2
            )
        return transition(reward=r1, observation=obs1), transition(
            reward=r2, observation=obs2
        )

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, theta1, theta2):
        th1, th2 = jnp.clip(theta1, 0, 1), jnp.clip(theta2, 0, 1)

        # actions are egocentric so swap around for player two
        _switch = jnp.array(
            [
                [
                    [1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            ]
            * th1.shape[0]
        )
        _th2 = jnp.einsum("Bik,Bk -> Bi", _switch, th2)
        v1, v2 = self._step_2(th1, _th2)

        # reward
        # actions are egocentric so swap around for player two
        # theta2 = jnp.einsum("Bik,Bk -> Bi", switch, theta2)
        # P = jnp.concatenate(
        #     [
        #         theta1 * theta2,
        #         theta1 * (1 - theta2),
        #         (1 - theta1) * theta2,
        #         (1 - theta1) * (1 - theta2),
        #     ],
        #     axis=-1,
        # )
        # # initial distibution over start p0
        # # P = jnp.einsum("Bji -> Bij", P)
        # P = jnp.reshape(P, (P.shape[0], 4, 5))
        # p0 = P[:, :, 5]
        # P = jnp.einsum("Bji-> Bij", P[:, :, :4])
        # inf_sum = jnp.linalg.inv(jnp.eye(4) - P * gamma)
        # v1 = jnp.einsum("Bi, Bij, Bj -> B", p0, inf_sum, payoff1)
        # v2 = jnp.einsum("Bi, Bij, Bj -> B", p0, inf_sum, payoff2)

        obs1 = jnp.concatenate([th1, th2], axis=1)
        obs2 = jnp.concatenate([th2, th1], axis=1)
        return v1, v2, obs1, obs2

    # TODO: for comparison on differntiation

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(num_values=10, name="previous policy")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.BoundedArray(
            shape=(self.num_envs, 5),
            minimum=0,
            maximum=1,
            name="action",
            dtype=float,
        )

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        self._num_steps = 0
        # TODO: unsure if this correct def
        obs = 0.5 * jnp.ones((self.num_envs, 10))
        return restart(obs), restart(obs)


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

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        obs_1, obs_2 = self._observation(
            State.START * jnp.ones((self.num_envs,))
        )
        self._num_steps = 0
        return restart(obs_1), restart(obs_2)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, a1, a2) -> Tuple[jnp.array, jnp.array]:
        """Returns the rewards of a step"""
        # Example payoffs
        #             CC      DC     CD     DD
        # IPD       = [[2,2], [3,0], [0,3], [1,1]]
        # Stag hunt = [[4,4], [3,1], [1,3], [2,2]]
        # BotS      = [[3,2], [0,0], [0,0], [2,3]]
        # Chicken   = [[0,0], [1,-1],[-1,1],[-2,-2]]
        cc_p1 = self.payoff[0][0] * (a1 - 1.0) * (a2 - 1.0)
        cc_p2 = self.payoff[0][1] * (a1 - 1.0) * (a2 - 1.0)

        dc_p1 = self.payoff[1][0] * a1 * (1.0 - a2)
        dc_p2 = self.payoff[1][1] * a1 * (1.0 - a2)

        cd_p1 = self.payoff[2][0] * (1.0 - a1) * a2
        cd_p2 = self.payoff[2][1] * (1.0 - a1) * a2

        dd_p1 = self.payoff[3][0] * a1 * a2
        dd_p2 = self.payoff[3][1] * a1 * a2

        r1 = cc_p1 + dc_p1 + cd_p1 + dd_p1
        r2 = cc_p2 + dc_p2 + cd_p2 + dd_p2

        return r1, r2

    @partial(jax.jit, static_argnums=(0,))
    def _get_state(self, a1: jnp.array, a2: jnp.array) -> jnp.array:
        return (
            0 * (1 - a1) * (1 - a2)
            + 1 * (a1) * (1 - a2)
            + 2 * (1 - a1) * (a2)
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

        return jax.nn.one_hot(o_state, len(State)), jax.nn.one_hot(
            state, len(State)
        )


class IteratedPrisonersDilemma(SequentialMatrixGame):
    """Define iterated prisoner's dilemma"""

    def __init__(self, episode_length: int, num_envs: int) -> None:
        # (CC, DC, CD, DD)
        self.payoff = jnp.array([[2, 2], [3, 0], [0, 3], [1, 1]])
        super().__init__(episode_length, num_envs, self.payoff)


class StagHunt(SequentialMatrixGame):
    """Define stag hunt"""

    def __init__(self, episode_length: int, num_envs: int) -> None:
        # (CC, DC, CD, DD)
        self.payoff = jnp.array([[4, 4], [3, 1], [1, 3], [2, 2]])
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
        self.payoff = jnp.array([[0, 0], [1, -1], [-1, 1], [-2, -2]])
        super().__init__(episode_length, num_envs, self.payoff)


if __name__ == "__main__":
    env = SequentialMatrixGame(
        5, 1, jnp.array([[2, 2], [3, 0], [0, 3], [1, 1]])
    )
