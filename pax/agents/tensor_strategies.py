from functools import partial
from re import A
from typing import Callable, NamedTuple

import jax.numpy as jnp
import jax.random
from pax.agents.agent import AgentInterface
from pax.agents.strategies import initial_state_fun

from pax.utils import Logger, MemoryState, TrainingState


class TitForTatStrictStay(AgentInterface):
    # Switch to what opponents did if the other two played the same move
    # otherwise play as before
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
        return self._reciprocity(obs), state, mem

    def _reciprocity(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        obs = obs.argmax(axis=-1)
        # in state 0-3 we cooped, 4-7 we defected
        # 0 is cooperate, 1 is defect
        # default is we play the same as before
        action = jnp.where(obs > 3, 1, 0)
        # if obs is 0 or 4, they both cooperated and we cooperate next round
        # if obs is 8, we start by cooperating
        action = jnp.where(obs % 4 == 0, 0, action)
        # if obs is 3 or 7, they both defected and we defect next round
        action = jnp.where(obs % 4 == 3, 1, action)
        return action


class TitForTatStrictSwitch(AgentInterface):
    # Switch to what opponents did if the other two played the same move
    # otherwise switch
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
        return self._reciprocity(obs), state, mem

    def _reciprocity(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        obs = obs.argmax(axis=-1)
        # in state 0-3 we cooped, 4-7 we defected
        # 0 is cooperate, 1 is defect
        # default is switch from before
        action = jnp.where(obs > 3, 0, 1)
        # if obs is 0 or 4, they both cooperated and we cooperate next round
        # if obs is 8, we start by cooperating
        action = jnp.where(obs % 4 == 0, 0, action)
        # if obs is 3 or 7, they both defected and we defect next round
        action = jnp.where(obs % 4 == 3, 1, action)
        return action
