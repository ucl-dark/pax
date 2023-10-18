from functools import partial
from typing import NamedTuple

import jax.numpy as jnp
import jax.random

from pax.agents.agent import AgentInterface
from pax.agents.strategies import initial_state_fun
from pax.utils import Logger, MemoryState


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


class TitForTatCooperate(AgentInterface):
    # Cooperate unless they both defect
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
        # in state 3 and 7 they both defected, we defect
        # otherwise we cooperate
        # 0 is cooperate, 1 is defect
        action = jnp.where(obs % 4 == 3, 1, 0)
        return action


class TitForTatDefect(AgentInterface):
    # Defect unless they both cooperate
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
        # in state 0 and 4 they both cooperated, we cooperate
        # state 8 is start, we cooperate
        # otherwise we defect
        # 0 is cooperate, 1 is defect
        action = jnp.where(obs % 4 == 0, 0, 1)
        return action


class TitForTatHarsh(AgentInterface):
    # Defect unless everyone cooperates, works for n players
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
        num_players = jnp.log2(obs.shape[-1] - 1).astype(int)
        obs = obs.argmax(axis=-1)

        # 0th state is all c...cc
        # 1st state is all c...cd
        # (2**num_playerth)-1 state is d...dd
        # we cooperate in states _CCCC (0th and (2**num_player-1)th)
        # and in start state (state 2**num_player)th ) otherwise we defect
        # 0 is cooperate, 1 is defect
        action = jnp.where(obs % jnp.exp2(num_players - 1) == 0, 0, 1)
        return action


class TitForTatSoft(AgentInterface):
    # Defect if majority defects, works for n players
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
        num_players = jnp.log2(obs.shape[-1] - 1).astype(jnp.int32)
        max_defect_allowed = (num_players - 1) / 2
        # from one_hot to int
        obs = jnp.argmax(obs, axis=-1, keepdims=True).astype(jnp.uint8)
        # convert to binary to get actions of opponents
        # -num_player-1th is the start state indicator not action
        obs_actions = jnp.unpackbits(obs, axis=-1)
        num_defects = obs_actions.sum(axis=-1)
        # substract our own actions and make sure start state has 0 defect
        # our actions are on obs_actions[...,-num_players]
        # start state is on obs_actions[...,-num_players-1]
        opps_defect = (
            num_defects
            - obs_actions[..., -num_players]
            - obs_actions[..., -num_players - 1]
        )
        # if more than half of them are 1, we defect
        # num_defects = jnp.sum(cutoff_obs_actions,axis=-1)
        assert opps_defect.shape == obs.shape[:-1]
        action = jnp.where(opps_defect <= max_defect_allowed, 0, 1).astype(
            jnp.int32
        )
        return action
