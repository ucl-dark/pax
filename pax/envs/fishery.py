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
    def __init__(self, num_players: int, num_inner_steps: int):
        super().__init__()
        self.num_players = num_players

        def _step(
                key: chex.PRNGKey,
                state: EnvState,
                actions: Tuple[float, ...],
                params: EnvParams,
        ):
            t = state.inner_t
            key, _ = jax.random.split(key, 2)
            done = t >= num_inner_steps

            actions = jnp.asarray(actions).squeeze()
            actions = jnp.clip(actions, a_min=0)
            E = actions.sum()
            s_growth = state.s + params.g * state.s * (1 - state.s / params.s_max)

            # Prevent s from dropping below 0
            H = jnp.clip(E * state.s * params.e, a_max=s_growth)
            s_next = s_growth - H
            next_state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t,
                s=s_next
            )
            reset_obs, reset_state = _reset(key, params)
            reset_state = reset_state.replace(outer_t=state.outer_t + 1)

            all_obs = []
            all_rewards = []
            for i in range(num_players):
                obs = jnp.concatenate([actions, jnp.array([s_next])])
                obs = jax.lax.select(done, reset_obs[i], obs)
                all_obs.append(obs)

                e = actions[i]
                # reward = benefit - cost
                # = P * H - w * E
                r = jnp.where(E != 0, params.P * e / E * H - params.w * e, 0)
                all_rewards.append(r)

            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y),
                reset_state,
                next_state,
            )

            return (
                tuple(all_obs),
                state,
                tuple(all_rewards),
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
            obs = jax.random.uniform(key, (num_players,))
            obs = jnp.concatenate([obs, jnp.array([state.s])])
            return tuple([obs for _ in range(num_players)]), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @staticmethod
    def name() -> str:
        """Environment name."""
        return "Fishery"

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
        return spaces.Box(low=0, high=float('inf'), shape=self.num_players + 1, dtype=jnp.float32)

    @staticmethod
    def equilibrium(params: EnvParams) -> float:
        return params.s_max * (1 - params.g / params.e / params.P)
