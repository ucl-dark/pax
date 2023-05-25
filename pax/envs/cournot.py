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
    a: int
    b: float
    marginal_cost: float


class CournotGame(environment.Environment):
    def __init__(self, num_steps: int):
        super().__init__()

        # TODO handle overproduction
        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[float, float],
            params: EnvParams,
        ):
            t = state.outer_t
            key, _ = jax.random.split(key, 2)
            q1 = actions[0]
            q2 = actions[1]
            p = params.a - params.b * (q1 + q2)
            r1 = jnp.squeeze(p * q1 - params.marginal_cost * q1)
            r2 = jnp.squeeze(p * q2 - params.marginal_cost * q2)
            obs1 = jnp.concatenate([q2, p])
            obs2 = jnp.concatenate([q1, p])

            done = t >= num_steps
            state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t + 1
            )
            return (
                (obs1, obs2),
                state,
                (r1, r2),
                done,
                # TODO needed?
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            obs = jax.random.uniform(key, (2,))
            return (obs, obs), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "CournotGame-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=0, high=float('inf'), shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=0, high=float('inf'), shape=2, dtype=jnp.float32)
