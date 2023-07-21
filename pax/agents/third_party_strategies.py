from functools import partial
from re import A
from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.random

from pax.agents.agent import AgentInterface
from pax.agents.strategies import initial_state_fun
from pax.utils import Logger, MemoryState, TrainingState


class CooperateNoPunish(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return mem

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self,
        state: NamedTuple,
        obs: jnp.ndarray,
        mem: NamedTuple,
    ) -> tuple[jnp.ndarray, NamedTuple, NamedTuple]:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._coop(obs), state, mem

    def _coop(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        obs = obs.argmax(axis=-1)
        obs.shape
        action = jnp.zeros(obs.shape, dtype=jnp.int32)
        return action


class DefectPunish(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return mem

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self,
        state: NamedTuple,
        obs: jnp.ndarray,
        mem: NamedTuple,
    ) -> tuple[jnp.ndarray, NamedTuple, NamedTuple]:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._coop(obs), state, mem

    def _coop(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        obs = obs.argmax(axis=-1)
        obs.shape
        action = jnp.ones(obs.shape, dtype=jnp.int32) * 15
        return action
