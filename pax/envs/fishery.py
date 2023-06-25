from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int
    s: float


@chex.dataclass
class EnvParams:
    g: float
    e: float
    P: float
    w: float
    s_0: float
    s_max: float

def to_obs_array(params: EnvParams) -> jnp.ndarray:
    return jnp.array([params.g, params.e, params.P, params.w])


"""
Based off the open access fishery environment from Perman et al. (3rd edition)

Biological sub-model:

The stock of fish is given by G and grows at a natural rate of g. 
The stock is harvested at a rate of E (sum over agent actions) and the harvest is sold at a price of p. 

g      growth rate
e      efficiency parameter
P      price of fish
w      cost per unit of effort
s_0    initial stock TODO randomize?
s_max  maximum stock size
"""


class Fishery(environment.Environment):
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
            e1 = actions[0]
            e2 = actions[1]
            E = e1 + e2
            H = E * state.s * params.e
            s_next = state.s + params.g * state.s * (1 - state.s / params.s_max) - H

            # reward = benefit - cost
            # = P * H - w * E
            r1 = params.P * e1 * state.s * params.e - params.w * e1
            r2 = params.P * e2 * state.s * params.e - params.w * e2

            obs1 = jnp.concatenate([to_obs_array(params), jnp.array([state.s, e1, e2])])
            obs2 = jnp.concatenate([to_obs_array(params), jnp.array([state.s, e1, e2])])

            done = t >= num_inner_steps
            state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t + 1,
                s=s_next
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
                s=params.s_0
            )
            obs = jax.random.uniform(key, (2,))
            obs = jnp.concatenate([to_obs_array(params), jnp.array([state.s]), obs])
            return (obs, obs), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Fishery-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(
            self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=0, high=params.s_max, shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=0, high=float('inf'), shape=7, dtype=jnp.float32)
