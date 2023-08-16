from typing import Optional, Tuple

import chex
import jax
import jax.debug
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from pax.envs.rice import Rice, EnvState, EnvParams

"""
Wrapper to turn Rice into a single-agent environment.
"""


class SarlRice(environment.Environment):
    env_id: str = "SarlRice-v1"

    def __init__(self, num_inner_steps: int, config_folder: str):
        super().__init__()
        self.rice = Rice(num_inner_steps, config_folder)

        def _step(
                key: chex.PRNGKey,
                state: EnvState,
                action: chex.Array,
                params: EnvParams,
        ):
            actions = jnp.split(action, self.rice.num_players)
            obs, state, rewards, done, info = self.rice.step(key, state, tuple(actions), params)

            return (
                jnp.concatenate(obs),
                state,
                jnp.asarray(rewards).sum(),
                done,
                info,
            )

        def _reset(
                key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[chex.Array, EnvState]:
            obs, state = self.rice.reset(key, params)
            return jnp.asarray(obs), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        return self.env_id

    @property
    def num_actions(self) -> int:
        return self.rice.num_actions * self.rice.num_players

    def action_space(
            self, params: Optional[EnvParams] = None
    ) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=(self.num_actions,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        obs_space = self.rice.observation_space(params)
        obs_space.shape = (obs_space.shape[0] * self.rice.num_players,)
        return obs_space
