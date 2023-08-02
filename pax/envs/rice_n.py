from typing import Optional, Tuple

import chex
import jax
import jax.debug
import jax.numpy as jnp
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int

    activity_step: int

    # Ecological
    global_temp: chex.ArrayDevice
    global_carbon_mass: chex.ArrayDevice
    global_exogenous_emissions: float
    global_land_emissions: float

    # Economic
    labor_all_regions: chex.ArrayDevice
    production_factor_all_regions: chex.ArrayDevice
    intensity_all_regions: chex.ArrayDevice




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
Based off the MARL environment from https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4189735
which in turn is adapted from the RICE IAM
"""


class RiceN(environment.Environment):
    def __init__(self, num_inner_steps: int):
        super().__init__()

        def _step(
                key: chex.PRNGKey,
                state: EnvState,
                actions: Tuple[float, float],
                params: EnvParams,
        ):
            t = state.inner_t
            key, _ = jax.random.split(key, 2)

            done = t >= num_inner_steps

            next_state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t,
            )
            reset_obs, reset_state = _reset(key, params)
            reset_state = reset_state.replace(outer_t=state.outer_t + 1)

            obs1 = jnp.where(done, reset_obs[0], obs1)
            obs2 = jnp.where(done, reset_obs[1], obs2)

            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y),
                reset_state,
                next_state,
            )
            r1 = jax.lax.select(done, 0.0, r1)
            r2 = jax.lax.select(done, 0.0, r2)

            return (
                (obs1, obs2),
                state,
                (r1, r2),
                done,
                {
                    "H": H,
                    "E": E,
                },
            )

        def _reset(
                key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int16),
                outer_t=jnp.zeros((), dtype=jnp.int16),
                s=params.s_0
            )
            obs = jax.random.uniform(key, (2,))
            obs = jnp.concatenate([jnp.array([state.s]), obs, to_obs_array(params)])
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

    @staticmethod
    def equilibrium(params: EnvParams) -> float:
        return params.s_max * (1 - params.g / params.e / params.P)
