from functools import partial
from typing import Callable, NamedTuple, Union

import jax.numpy as jnp
import jax.random

from pax.agents.agent import AgentInterface
from pax.utils import Logger, MemoryState, TrainingState

# states are [CC, CD, DC, DD, START]
# actions are cooperate = 0 or defect = 1


def initial_state_fun(num_envs: int) -> Callable:
    def fun(key, hidden):
        return (
            TrainingState(None, None, jax.random.PRNGKey(0), None),
            MemoryState(
                hidden=jnp.zeros((num_envs, 1)),
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            ),
        )

    return fun


def reset_mem_fun(num_envs: int) -> Callable:
    def fun(memory, eval=False):
        memory = memory._replace(
            extras={
                "values": jnp.zeros(1 if eval else num_envs),
                "log_probs": jnp.zeros(1 if eval else num_envs),
            },
        )
        return memory

    return fun


class EvilGreedy(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self.reset_memory = reset_mem_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

        def _greedy_step(obs):
            """do this for a single obs and then vmap"""
            # reshape so that obs is no longer flattened
            obs = obs.reshape(3, 3, 4)

            # [3, 3]
            # [3, 3]
            agent_coin_pos = obs[..., 2]
            other_coin_pos = obs[..., 3]

            # find path to both sets of coins
            agent_loc = jnp.array([1, 1])
            # assumes square grid
            x, y = jnp.divmod(jnp.argmax(agent_coin_pos), 3)
            agent_path = (
                jnp.array(
                    [
                        x,
                        y,
                    ]
                )
                - agent_loc
            )
            x, y = jnp.divmod(jnp.argmax(other_coin_pos), 3)
            other_path = jnp.array([x, y]) - agent_loc

            agent_path_length = jnp.sum(jnp.abs(agent_path))
            other_path_length = jnp.sum(jnp.abs(other_path))
            path = jnp.where(
                agent_path_length < other_path_length, agent_path, other_path
            )

            stay = (jnp.array([0, 0]) == path).all()
            right = (jnp.array([0, 1]) == path).all()
            left = (jnp.array([0, -1]) == path).all()
            up = (jnp.array([1, 0]) == path).all()
            down = (jnp.array([-1, 0]) == path).all()

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

        self._greedy_step = _greedy_step
        greedy_step = jax.vmap(_greedy_step)

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:

            return greedy_step(obs), state, mem

        self._policy = _policy

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class RandomGreedy(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self.reset_memory = reset_mem_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

        def _greedy_step(obs, rng):
            """do this for a single obs and then vmap"""
            # reshape so that obs is no longer flattened
            obs = obs.reshape(3, 3, 4)

            # [3, 3]
            # [3, 3]
            agent_coin_pos = obs[..., 2]
            other_coin_pos = obs[..., 3]

            # find path to both sets of coins
            agent_loc = jnp.array([1, 1])
            # assumes square grid
            x, y = jnp.divmod(jnp.argmax(agent_coin_pos), 3)
            agent_path = (
                jnp.array(
                    [
                        x,
                        y,
                    ]
                )
                - agent_loc
            )
            x, y = jnp.divmod(jnp.argmax(other_coin_pos), 3)
            other_path = jnp.array([x, y]) - agent_loc

            agent_path_length = jnp.sum(jnp.abs(agent_path))
            other_path_length = jnp.sum(jnp.abs(other_path))

            equal_path = jnp.where(
                agent_path_length <= other_path_length, agent_path, other_path
            )
            new_rng, _ = jax.random.split(rng)
            p = jax.random.uniform(
                new_rng,
            )
            random_path = jnp.where(p < 0.5, agent_path, other_path)

            shorted_path = jnp.where(
                agent_path_length < other_path_length, agent_path, other_path
            )

            path = jnp.where(equal_path, random_path, shorted_path)

            stay = (jnp.array([0, 0]) == path).all()
            right = (jnp.array([0, 1]) == path).all()
            left = (jnp.array([0, -1]) == path).all()
            up = (jnp.array([1, 0]) == path).all()
            down = (jnp.array([-1, 0]) == path).all()

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
            return action, new_rng

        self._greedy_step = _greedy_step
        greedy_step = jax.vmap(
            _greedy_step, in_axes=(0, None), out_axes=(0, None)
        )

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:

            action, new_rng = greedy_step(obs, state.random_key)
            state = state._replace(random_key=new_rng)
            return action, state, mem

        self._policy = _policy

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class GoodGreedy(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self.reset_memory = reset_mem_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

        def _greedy_step(obs):
            """do this for a single obs and then vmap"""
            # reshape so that obs is no longer flattened
            obs = obs.reshape(3, 3, 4)

            # [3, 3]
            # [3, 3]
            agent_coin_pos = obs[..., 2]
            other_coin_pos = obs[..., 3]

            # find path to both sets of coins
            agent_loc = jnp.array([1, 1])
            # assumes square grid
            x, y = jnp.divmod(jnp.argmax(agent_coin_pos), 3)
            agent_path = (
                jnp.array(
                    [
                        x,
                        y,
                    ]
                )
                - agent_loc
            )
            x, y = jnp.divmod(jnp.argmax(other_coin_pos), 3)
            other_path = jnp.array([x, y]) - agent_loc

            agent_path_length = jnp.sum(jnp.abs(agent_path))
            other_path_length = jnp.sum(jnp.abs(other_path))
            path = jnp.where(
                agent_path_length <= other_path_length, agent_path, other_path
            )

            stay = (jnp.array([0, 0]) == path).all()
            right = (jnp.array([0, 1]) == path).all()
            left = (jnp.array([0, -1]) == path).all()
            up = (jnp.array([1, 0]) == path).all()
            down = (jnp.array([-1, 0]) == path).all()

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

        self._greedy_step = _greedy_step
        greedy_step = jax.vmap(_greedy_step)

        def _policy(
            state: NamedTuple,
            obs: jnp.array,
            mem: NamedTuple,
        ) -> jnp.ndarray:

            return greedy_step(obs), state, mem

        self._policy = _policy

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class GrimTrigger(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger
        self._logger.metrics = {}

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


class TitForTat(AgentInterface):
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
        obs: jnp.array,
        mem: NamedTuple,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._reciprocity(obs), state, mem

    def _reciprocity(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        # now either 0, 1, 2, 3
        # batch_size, _ = obs.shape
        obs = obs.argmax(axis=-1)
        # if 0 | 2 | 4  -> C
        # if 1 | 3 -> D
        obs = obs % 2
        action = jnp.where(obs > 0, 1, 0)
        return action


class Defect(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

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


class Altruistic(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

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


class Random(AgentInterface):
    def __init__(
        self, num_actions: int, num_envs: int, obs_shape: Union[dict, tuple]
    ):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self.reset_memory = reset_mem_fun(num_envs)
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
            if isinstance(obs_shape, dict):
                batch_size = obs["inventory"].shape[0]
            else:
                batch_size = obs.shape[0]
            new_key, _ = jax.random.split(state.random_key)
            action = jax.random.randint(new_key, (batch_size,), 0, num_actions)
            state = state._replace(random_key=new_key)
            return action, state, mem

        self._policy = jax.jit(_policy)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class Stay(AgentInterface):
    def __init__(self, num_actions: int, num_envs: int, num_players: int = 2):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
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
            action = 4 * jnp.ones((num_envs,), dtype=int)
            return action, state, mem

        self._policy = jax.jit(_policy)

    def update(self, unused0, unused1, state, mem) -> None:
        return state, mem, {}

    def reset_memory(self, mem, *args) -> MemoryState:
        return self._mem

    def make_initial_state(self, _unused, *args) -> TrainingState:
        return self._state, self._mem


class HyperAltruistic(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()

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


class HyperDefect(AgentInterface):
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

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
    def __init__(self, num_envs, *args):
        self.make_initial_state = initial_state_fun(num_envs)
        self._state, self._mem = self.make_initial_state(None, None)
        self._logger = Logger()
        self._logger.metrics = {}

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
