from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice


class IteratedTensorGame(environment.Environment):
    """
    JAX Compatible version of tensor game environment.
    """

    def __init__(self, num_inner_steps: int, num_outer_steps: int):
        super().__init__()

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int, int],
            params: EnvParams,
        ):
            inner_t, outer_t = state.inner_t, state.outer_t
            a1, a2, a3 = actions
            inner_t += 1

            ccc_p1 = (
                params.payoff_matrix[0][0]
                * (1.0 - a1)
                * (1.0 - a2)
                * (1.0 - a3)
            )
            ccc_p2 = (
                params.payoff_matrix[0][1]
                * (1.0 - a1)
                * (1.0 - a2)
                * (1.0 - a3)
            )
            ccc_p3 = (
                params.payoff_matrix[0][2]
                * (1.0 - a1)
                * (1.0 - a2)
                * (1.0 - a3)
            )
            ccd_p1 = params.payoff_matrix[1][0] * (1.0 - a1) * (1.0 - a2) * a3
            ccd_p2 = params.payoff_matrix[1][1] * (1.0 - a1) * (1.0 - a2) * a3
            ccd_p3 = params.payoff_matrix[1][2] * (1.0 - a1) * (1.0 - a2) * a3
            cdc_p1 = params.payoff_matrix[2][0] * (1.0 - a1) * a2 * (1.0 - a3)
            cdc_p2 = params.payoff_matrix[2][1] * (1.0 - a1) * a2 * (1.0 - a3)
            cdc_p3 = params.payoff_matrix[2][2] * (1.0 - a1) * a2 * (1.0 - a3)
            cdd_p1 = params.payoff_matrix[3][0] * (1.0 - a1) * a2 * (a3)
            cdd_p2 = params.payoff_matrix[3][1] * (1.0 - a1) * a2 * (a3)
            cdd_p3 = params.payoff_matrix[3][2] * (1.0 - a1) * a2 * (a3)
            dcc_p1 = (
                params.payoff_matrix[4][0] * (a1) * (1.0 - a2) * (1.0 - a3)
            )
            dcc_p2 = (
                params.payoff_matrix[4][1] * (a1) * (1.0 - a2) * (1.0 - a3)
            )
            dcc_p3 = (
                params.payoff_matrix[4][2] * (a1) * (1.0 - a2) * (1.0 - a3)
            )
            dcd_p1 = params.payoff_matrix[5][0] * (a1) * (1.0 - a2) * (a3)
            dcd_p2 = params.payoff_matrix[5][1] * (a1) * (1.0 - a2) * (a3)
            dcd_p3 = params.payoff_matrix[5][2] * (a1) * (1.0 - a2) * (a3)
            ddc_p1 = params.payoff_matrix[6][0] * (a1) * (a2) * (1.0 - a3)
            ddc_p2 = params.payoff_matrix[6][1] * (a1) * (a2) * (1.0 - a3)
            ddc_p3 = params.payoff_matrix[6][2] * (a1) * (a2) * (1.0 - a3)
            ddd_p1 = params.payoff_matrix[7][0] * (a1) * (a2) * (a3)
            ddd_p2 = params.payoff_matrix[7][1] * (a1) * (a2) * (a3)
            ddd_p3 = params.payoff_matrix[7][2] * (a1) * (a2) * (a3)

            r1 = (
                ccc_p1
                + ccd_p1
                + cdc_p1
                + cdd_p1
                + dcc_p1
                + dcd_p1
                + ddc_p1
                + ddd_p1
            )
            r2 = (
                ccc_p2
                + ccd_p2
                + cdc_p2
                + cdd_p2
                + dcc_p2
                + dcd_p2
                + ddc_p2
                + ddd_p2
            )
            r3 = (
                ccc_p3
                + ccd_p3
                + cdc_p3
                + cdd_p3
                + dcc_p3
                + dcd_p3
                + ddc_p3
                + ddd_p3
            )

            s1 = (
                0 * (1 - a1) * (1 - a2) * (1 - a3)
                + 1 * (1 - a1) * (1 - a2) * (a3)
                + 2 * (1 - a1) * (a2) * (1 - a3)
                + 3 * (1 - a1) * (a2) * (a3)
                + 4 * (a1) * (1 - a2) * (1 - a3)
                + 5 * (a1) * (1 - a2) * (a3)
                + 6 * (a1) * (a2) * (1 - a3)
                + 7 * (a1) * (a2) * (a3)
            )

            s2 = (
                0 * (1 - a2) * (1 - a3) * (1 - a1)
                + 1 * (1 - a2) * (1 - a3) * (a1)
                + 2 * (1 - a2) * (a3) * (1 - a1)
                + 3 * (1 - a2) * (a3) * (a1)
                + 4 * (a2) * (1 - a3) * (1 - a1)
                + 5 * (a2) * (1 - a3) * (a1)
                + 6 * (a2) * (a3) * (1 - a1)
                + 7 * (a2) * (a3) * (a1)
            )
            s3 = (
                0 * (1 - a3) * (1 - a1) * (1 - a2)
                + 1 * (1 - a3) * (1 - a1) * (a2)
                + 2 * (1 - a3) * (a1) * (1 - a2)
                + 3 * (1 - a3) * (a1) * (a2)
                + 4 * (a3) * (1 - a1) * (1 - a2)
                + 5 * (a3) * (1 - a1) * (a2)
                + 6 * (a3) * (a1) * (1 - a2)
                + 7 * (a3) * (a1) * (a2)
            )
            # if first step then return START state.
            reset_inner = inner_t == num_inner_steps
            s1 = jax.lax.select(reset_inner, jnp.int8(8), jnp.int8(s1))
            s2 = jax.lax.select(reset_inner, jnp.int8(8), jnp.int8(s2))
            s3 = jax.lax.select(reset_inner, jnp.int8(8), jnp.int8(s3))

            obs1 = jax.nn.one_hot(s1, 9, dtype=jnp.int8)
            obs2 = jax.nn.one_hot(s2, 9, dtype=jnp.int8)
            obs3 = jax.nn.one_hot(s3, 9, dtype=jnp.int8)

            # out step keeping
            inner_t = jax.lax.select(
                reset_inner, jnp.zeros_like(inner_t), inner_t
            )
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            reset_outer = outer_t == num_outer_steps
            state = EnvState(inner_t=inner_t, outer_t=outer_t)
            return (
                (obs1, obs2, obs3),
                state,
                (r1, r2, r3),
                reset_outer,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            obs = jax.nn.one_hot(8 * jnp.ones(()), 9, dtype=jnp.int8)
            return (obs, obs, obs), state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "IteratedTensorGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    # def action_space(
    #     self, params: Optional[EnvParams] = None
    # ) -> spaces.Discrete:
    #     """Action space of the environment."""
    #     return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Discrete(9)

    # def state_space(self, params: EnvParams) -> spaces.Dict:
    #     """State space of the environment."""
    #     return spaces.Discrete(5)
