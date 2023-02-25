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


class IteratedMatrixGame(environment.Environment):
    """
    JAX Compatible version of matrix game environment.
    """

    def __init__(self, num_inner_steps: int, num_outer_steps: int):
        super().__init__()

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[int, int],
            params: EnvParams,
        ):
            inner_t, outer_t = state.inner_t, state.outer_t
            a1, a2 = actions
            inner_t += 1

            cc_p1 = params.payoff_matrix[0][0] * (a1 - 1.0) * (a2 - 1.0)
            cc_p2 = params.payoff_matrix[0][1] * (a1 - 1.0) * (a2 - 1.0)
            cd_p1 = params.payoff_matrix[1][0] * (1.0 - a1) * a2
            cd_p2 = params.payoff_matrix[1][1] * (1.0 - a1) * a2
            dc_p1 = params.payoff_matrix[2][0] * a1 * (1.0 - a2)
            dc_p2 = params.payoff_matrix[2][1] * a1 * (1.0 - a2)
            dd_p1 = params.payoff_matrix[3][0] * a1 * a2
            dd_p2 = params.payoff_matrix[3][1] * a1 * a2

            r1 = cc_p1 + dc_p1 + cd_p1 + dd_p1
            r2 = cc_p2 + dc_p2 + cd_p2 + dd_p2

            s1 = (
                0 * (1 - a1) * (1 - a2)
                + 1 * (1 - a1) * a2
                + 2 * a1 * (1 - a2)
                + 3 * (a1) * (a2)
            )

            s2 = (
                0 * (1 - a1) * (1 - a2)
                + 1 * a1 * (1 - a2)
                + 2 * (1 - a1) * a2
                + 3 * (a1) * (a2)
            )
            # if first step then return START state.
            reset_inner = inner_t == num_inner_steps + 1
            s1 = jax.lax.select(reset_inner, jnp.int8(4), jnp.int8(s1))
            s2 = jax.lax.select(reset_inner, jnp.int8(4), jnp.int8(s2))
            obs1 = jax.nn.one_hot(s1, 5, dtype=jnp.int8)
            obs2 = jax.nn.one_hot(s2, 5, dtype=jnp.int8)

            # out step keeping
            inner_t = jax.lax.select(
                reset_inner, jnp.zeros_like(inner_t), inner_t
            )
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            reset_outer = outer_t == num_outer_steps
            state = EnvState(inner_t=inner_t, outer_t=outer_t)
            return (
                (obs1, obs2),
                state,
                (r1, r2),
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
            obs = jax.nn.one_hot(4 * jnp.ones(()), 5, dtype=jnp.int8)
            return (obs, obs), state

        # overwrite Gymnax as it makes single-agent assumptions
        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "IteratedMatrixGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(4)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Discrete(5)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Discrete(5)
