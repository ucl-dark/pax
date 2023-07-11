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


def to_obs_array(params: EnvParams) -> jnp.ndarray:
    return jnp.array([params.a, params.b, params.marginal_cost])


class CournotGame(environment.Environment):
    def __init__(self, num_inner_steps: int):
        super().__init__()

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
            obs1 = jnp.concatenate([to_obs_array(params), jnp.array(q2), jnp.array(p)])
            obs2 = jnp.concatenate([to_obs_array(params), jnp.array(q1), jnp.array(p)])

            done = t >= num_inner_steps
            state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t + 1
            )
            return (
                (obs1, obs2),
                state,
                (r1, r2),
                done,
                {},
            )

        def _reset(
                key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            obs = jax.random.uniform(key, (2,))
            obs = jnp.concatenate([to_obs_array(params), obs])
            return (obs, obs), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Cournot-v1"

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
        return spaces.Box(low=0, high=float('inf'), shape=5, dtype=jnp.float32)

    @staticmethod
    def nash_policy(params: EnvParams) -> float:
        return 2 * (params.a - params.marginal_cost) / (3 * params.b)

    @staticmethod
    def nash_reward(params: EnvParams) -> float:
        q = CournotGame.nash_policy(params)
        p = params.a - params.b * q
        return p * q - params.marginal_cost * q
