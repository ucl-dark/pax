from functools import partial
from typing import Mapping, NamedTuple, Tuple

import jax.numpy as jnp
import jax.random
from dm_env import TimeStep

from pax.utils import MemoryState, TrainingState

# states are [CC, CD, DC, DD, START]
# actions are cooperate = 0 or defect = 1


class GrimTrigger:
    def __init__(self, *args):
        def make_initial_state(key, hidden):
            return TrainingState(
                None, None, jax.random.PRNGKey(0), None
            ), MemoryState(jnp.ones(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        return self._trigger(timestep.observation)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem

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
        def make_initial_state(key, hidden):
            return TrainingState(
                None, None, jax.random.PRNGKey(0), None
            ), MemoryState(jnp.ones(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._reciprocity(timestep.observation)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem

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
        # action = jnp.expand_dims(action, axis=-1) # removing this in preference for (action, )
        return action


class Defect:
    def __init__(self, *args):
        def make_initial_state(key, hidden):
            return TrainingState(
                None, None, jax.random.PRNGKey(0), None
            ), MemoryState(jnp.zeros(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

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
        return state, mem

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
        def make_initial_state(key, hidden):
            return TrainingState(
                None, None, jax.random.PRNGKey(0), None
            ), MemoryState(jnp.zeros(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

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
        return state, mem

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
    def __init__(self, seed: int):
        self.rng = jax.random.PRNGKey(seed)

    def actor_step(self, timestep: TimeStep) -> Tuple[jnp.ndarray, None]:
        batch_size, _ = timestep.observation.shape
        # return jax.random.choice(self.rng, 2, [batch_size, 1]), None
        return (
            jax.random.choice(
                self.rng,
                2,
                [
                    batch_size,
                ],
            ),
            None,
        )


class HyperAltruistic:
    def __init__(self, *args):
        def make_initial_state(key, hidden):
            return TrainingState(
                None, None, jax.random.PRNGKey(0), None
            ), MemoryState(jnp.zeros(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

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
        return state, mem

    def reset_memory(self, *args) -> TrainingState:
        return self._mem


class HyperDefect:
    def __init__(self, *args):
        def make_initial_state(key, hidden):
            return TrainingState(
                None, jax.random.PRNGKey(0), None, None
            ), MemoryState(jnp.zeros(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

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
        return state, mem

    def reset_memory(self, *args) -> TrainingState:
        return self._mem

    def make_initial_state(
        self, _unused, *args
    ) -> Tuple[TrainingState, MemoryState]:
        return self._state, self._mem


class HyperTFT:
    def __init__(self, *args):
        def make_initial_state(key, hidden):
            return TrainingState(
                None, None, jax.random.PRNGKey(0), None
            ), MemoryState(jnp.zeros(1), {"log_probs": None, "values": None})

        self.make_initial_state = make_initial_state
        self._state, self._mem = make_initial_state(None, None)

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
        return state, mem

    def reset_memory(self, *args) -> TrainingState:
        return self._mem
