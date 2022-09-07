from functools import partial
from time import time
from typing import Mapping, NamedTuple, Tuple

import jax.numpy as jnp
import jax.random
from dm_env import TimeStep

from pax.utils import MemoryState, TrainingState, Logger

# states are [CC, CD, DC, DD, START]
# actions are cooperate = 0 or defect = 1


def make_initial_state(key, hidden):
    return TrainingState(None, None, jax.random.PRNGKey(0), None), MemoryState(
        jnp.ones(1), {"log_probs": None, "values": None}
    )


class GreedyCoinChaser:
    def __init__(self, num_actions: int):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}
        self._num_actions = num_actions

        def _greedy_step(obs):
            """do this for a single obs and then vmap"""
            # reshape so that obs is no longer flattened
            obs = obs.reshape(3, 3, 4)

            # [3, 3]
            # [3, 3]
            agent_pos = obs[..., 0]
            agent_coin_pos = obs[..., 2]
            other_coin_pos = obs[..., 3]

            # find path to both sets of coins
            print(f"agent coin pos {agent_coin_pos}")
            agent_loc = jnp.array(
                [jnp.argmax(agent_pos[0]), jnp.argmax(agent_pos[1])]
            )
            agent_path = (
                jnp.array(
                    [
                        jnp.argmax(agent_coin_pos[0]),
                        jnp.argmax(agent_coin_pos[1]),
                    ]
                )
                - agent_loc
            )
            other_path = (
                jnp.array(
                    [
                        jnp.argmax(other_coin_pos[0]),
                        jnp.argmax(other_coin_pos[1]),
                    ]
                )
                - agent_loc
            )

            agent_path_length = jnp.sum(jnp.abs(agent_path))
            other_path_length = jnp.sum(jnp.abs(other_path))

            path = jnp.where(
                agent_path_length < other_path_length, agent_path, other_path
            )

            stay = (jnp.array([0, 0]) == path).all()
            right = (jnp.array([0, 1]) == path).all()
            left = (jnp.array([0, -1]) == path).all()
            up = (jnp.array([1, 0]) == path).all()
            down = (jnp.array([0, 1]) == path).all()

            ur = (jnp.array([1, 1]) == path).all()
            ul = (jnp.array([1, -1]) == path).all()
            dr = (jnp.array([-1, 1]) == path).all()
            dl = (jnp.array([-1, -1]) == path).all()

            action = 0  # default right
            action = jax.lax.select(stay, 4, action)
            action = jax.lax.select(right, 0, action)
            action = jax.lax.select(left, 1, action)
            action = jax.lax.select(up, 2, action)
            action = jax.lax.select(down, 3, action)

            # ul -> l ur -> u,  dl -> d, dr -> r,
            action = jax.lax.select(ur, 2, action)
            action = jax.lax.select(ul, 1, action)
            action = jax.lax.select(dr, 0, action)
            action = jax.lax.select(dl, 3, action)
            return action

        greedy_step = jax.vmap(_greedy_step)

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:

            return greedy_step(obs), state, mem

        self._policy = _policy

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        action, self._state, self._mem = self._policy(
            self._state, timestep.observation, self._mem
        )
        return action

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class GrimTrigger:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger
        self._logger.metrics = {}

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        return self._trigger(timestep.observation)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, *args) -> TrainingState:
        return self._state

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self,
        state: NamedTuple,
        obs: jnp.array,
        mem: NamedTuple,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._trigger(obs), state, mem

    def _trigger(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        # batch_size, _ = obs.shape
        obs = obs.argmax(axis=-1)
        # if 0 | 4 -> C
        # if 1 | 2 | 3 | -> D
        obs = obs % 4  # now either 0, 1, 2, 3
        action = jnp.where(obs > 0.0, 1.0, 0.0)
        return action


class TitForTat:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._reciprocity(timestep.observation)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return mem

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self,
        state: NamedTuple,
        obs: jnp.array,
        mem: NamedTuple,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._reciprocity(obs), state, mem

    def _reciprocity(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        # now either 0, 1, 2, 3
        batch_size, _ = obs.shape
        obs = obs.argmax(axis=-1)
        # if 0 | 2 | 4  -> C
        # if 1 | 3 -> D
        obs = obs % 2
        action = jnp.where(obs > 0.0, 1.0, 0.0)
        return action


class Defect:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        batch_size, _ = timestep.observation.shape
        # return jnp.ones((batch_size, 1))
        return jnp.ones((batch_size,))

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self,
        state: jnp.array,
        obs: jnp.array,
        mem: None,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        batch_size = obs.shape[0]
        return jnp.ones((batch_size,)), state, mem


class Altruistic:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = timestep.observation.shape
        # return jnp.zeros((batch_size, 1))
        return jnp.zeros((batch_size,))

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self,
        state: NamedTuple,
        obs: jnp.array,
        mem: NamedTuple,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        batch_size = obs.shape[0]
        return jnp.zeros((batch_size,)), state, mem

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class Human:
    def __init__(self, *args):
        pass

    def select_action(self, timestep: TimeStep) -> Tuple[jnp.ndarray, None]:
        text = None
        batch_size, _ = timestep.observation.shape

        while text not in ("0", "1"):
            text = input("Press 0 for cooperate, Press 1 for defect:\n")

        # return int(text) * jnp.ones((batch_size, 1)), None
        return int(text) * jnp.ones((batch_size,)), None

    def step(self, *args) -> None:
        pass


class Random:
    def __init__(self, num_actions: int):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}
        self._num_actions = num_actions

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:
            # state is [batch x time_step x num_players]
            # return [batch]
            batch_size = obs.shape[0]
            new_key, _ = jax.random.split(state.random_key)
            action = jax.random.randint(new_key, (batch_size,), 0, num_actions)
            state = state._replace(random_key=new_key)
            return action, state, mem

        self._policy = jax.jit(_policy)

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = timestep.observation.shape
        action = jax.random.randint(
            self._state.new_key, (batch_size,), 0, self._num_actions
        )
        return action

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class HyperAltruistic:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()

    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = timestep.observation.shape
        return 20 * jnp.ones((batch_size, 5))

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self, state: NamedTuple, observation: jnp.array, mem: NamedTuple
    ) -> jnp.ndarray:
        (
            batch_size,
            _,
        ) = observation.shape
        action = jnp.tile(20 * jnp.ones((5,)), (batch_size, 1))
        return action, state, mem

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, *args) -> TrainingState:
        return self._mem


class HyperDefect:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = timestep.observation.shape
        return -20 * jnp.ones((batch_size, 5))

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self, state: NamedTuple, observation: jnp.array, mem: NamedTuple
    ) -> jnp.ndarray:
        (
            batch_size,
            _,
        ) = observation.shape
        action = jnp.tile(-20 * jnp.ones((5,)), (batch_size, 1))
        return action, state, mem

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, *args) -> TrainingState:
        return self._mem


class HyperTFT:
    def __init__(self, *args):
        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = timestep.observation.shape
        # return jnp.zeros((batch_size, 1))
        return jnp.tile(
            20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (batch_size, 1)
        )

    @partial(jax.jit, static_argnums=(0,))
    def _policy(
        self, state: NamedTuple, observation: jnp.array, mem: NamedTuple
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = observation.shape
        # return jnp.zeros((batch_size, 1))
        action = jnp.tile(
            20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (batch_size, 1)
        )
        return action, state, mem

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, *args) -> TrainingState:
        return self._mem
