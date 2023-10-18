from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    gamma: float


class InfiniteMatrixGame(environment.Environment):
    def __init__(self, num_steps: int):
        super().__init__()

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            # Thank you @luchris429 for this code!
            t = state.outer_t
            key, _ = jax.random.split(key, 2)
            payout_mat_1 = jnp.array([[r[0] for r in params.payoff_matrix]])
            payout_mat_2 = jnp.array([[r[1] for r in params.payoff_matrix]])

            theta1, theta2 = actions
            theta1, theta2 = jax.nn.sigmoid(theta1), jax.nn.sigmoid(theta2)
            obs1 = jnp.concatenate([theta1, theta2])
            obs2 = jnp.concatenate([theta2, theta1])

            _th2 = jnp.array(
                [theta2[0], theta2[2], theta2[1], theta2[3], theta2[4]]
            )

            p_1_0 = theta1[4:5]
            p_2_0 = _th2[4:5]
            p = jnp.concatenate(
                [
                    p_1_0 * p_2_0,
                    p_1_0 * (1 - p_2_0),
                    (1 - p_1_0) * p_2_0,
                    (1 - p_1_0) * (1 - p_2_0),
                ]
            )
            p_1 = jnp.reshape(theta1[0:4], (4, 1))
            p_2 = jnp.reshape(_th2[0:4], (4, 1))
            P = jnp.concatenate(
                [
                    p_1 * p_2,
                    p_1 * (1 - p_2),
                    (1 - p_1) * p_2,
                    (1 - p_1) * (1 - p_2),
                ],
                axis=1,
            )
            M = jnp.matmul(p, jnp.linalg.inv(jnp.eye(4) - params.gamma * P))
            L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (4, 1)))
            L_2 = jnp.matmul(M, jnp.reshape(payout_mat_2, (4, 1)))
            r1 = (1 - params.gamma) * L_1.sum()
            r2 = (1 - params.gamma) * L_2.sum()
            done = t >= num_steps
            state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t + 1
            )
            return (
                (obs1, obs2),
                state,
                (r1, r2),
                done,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            obs = jax.nn.sigmoid(jax.random.uniform(key, (10,)))
            return (obs, obs), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "InfiniteMatrixGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 5

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=0, high=1, shape=(5,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, (10,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Box:
        """State space of the environment."""
        return spaces.Box(0, 1, (10,), dtype=jnp.float32)
