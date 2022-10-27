import jax
from flax import struct
import jax.numpy as jnp
import chex
from gymnax.environments import environment, spaces
from typing import Tuple, Optional


class EnvState(struct.DataClass):
    inner_t: int
    outer_t: int


class EnvParams(struct.DataClass):
    payoff_matrix: jnp.ndarray
    num_inner_steps: int
    num_outer_steps: int
    num_players: int
    num_actions: int


class IteratedMatrixGame(environment.Environment):
    """
    JAX Compatible version of matrix game environment. Source:
    """

    def __init__(self):
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

            cc_p1 = params.payoff[0][0] * (a1 - 1.0) * (a2 - 1.0)
            cc_p2 = params.payoff[0][1] * (a1 - 1.0) * (a2 - 1.0)
            cd_p1 = params.payoff[1][0] * (1.0 - a1) * a2
            cd_p2 = params.payoff[1][1] * (1.0 - a1) * a2
            dc_p1 = params.payoff[2][0] * a1 * (1.0 - a2)
            dc_p2 = params.payoff[2][1] * a1 * (1.0 - a2)
            dd_p1 = params.payoff[3][0] * a1 * a2
            dd_p2 = params.payoff[3][1] * a1 * a2

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
            done = inner_t % params.num_inner_steps == 0
            s1 = jax.lax.select(done, jnp.int8(4), jnp.int8(s1))
            s2 = jax.lax.select(done, jnp.int8(4), jnp.int8(s2))
            obs1 = jax.nn.one_hot(s1, 5, dtype=jnp.int8)
            obs2 = jax.nn.one_hot(s2, 5, dtype=jnp.int8)

            # out step keeping
            reset_inner = inner_t == params.num_inner_steps
            inner_t = jax.lax.select(
                reset_inner, jnp.zeros_like(inner_t), inner_t
            )
            outer_t_new = outer_t + 1
            outer_t = jax.lax.select(reset_inner, outer_t_new, outer_t)
            discount = jnp.zeros((), dtype=jnp.int8)
            state = EnvState(inner_t, outer_t)
            return (obs1, obs2), state, (r1, r2), done, {"discount": discount}

        # for runner
        # self.runner_step = jax.jit(jax.vmap(_step))
        # self.batch_step = jax.jit(jax.vmap(jax.vmap(_step)))
        # self.runner_reset = runner_reset

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

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        state = EnvState(
            jnp.zeros((), dtype=jnp.int8),
            jnp.zeros((), dtype=jnp.int8),
        )
        obs = jax.nn.one_hot(4 * jnp.ones(()), 5, dtype=jnp.int8)
        return (obs, obs), state
