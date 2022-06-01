from functools import partial
from typing import Tuple
from chex import PRNGKey
import wandb

import jax.numpy as jnp
import jax.random
from dm_env import TimeStep

# states are [CC, DC, CD, DD]
# actions are cooperate = 0 or defect = 1

class TitForTat:
    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        key, 
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x time_step x num_players]
        # return [batch]
        return self._reciprocity(timestep.observation)

    def update(self, *args) -> None:
        pass

    def _reciprocity(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        # now either 0, 1, 2, 3
        batch_size, _ = obs.shape
        obs = obs.argmax(axis=-1)
        # if 0 | 1 | 4  -> C
        # if 2 | 3 -> D
        obs = obs % 4
        action = jnp.where(obs > 1.0, 1.0, 0.0)
        action = jnp.expand_dims(action, axis=-1)
        return action

class Defect:
    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self, 
        key,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        batch_size, _ = timestep.observation.shape
        return jnp.ones((batch_size, 1))

    def update(self, *args) -> None:
        pass

class Altruistic:
    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        key, 
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        (
            batch_size,
            _,
        ) = timestep.observation.shape
        return jnp.zeros((batch_size, 1))

    def update(self, *args) -> None:
        pass


class Human:
    def select_action(self, timestep: TimeStep) -> Tuple[jnp.ndarray, None]:
        text = None
        batch_size, _ = timestep.observation.shape

        while text not in ("0", "1"):
            text = input("Press 0 for cooperate, Press 1 for defect:\n")

        return int(text) * jnp.ones((batch_size, 1)), None

    def step(self, *args) -> None:
        pass


class Random:
    def __init__(self, seed: int):
        self.rng = jax.random.PRNGKey(seed)

    def actor_step(self, timestep: TimeStep) -> Tuple[jnp.ndarray, None]:
        batch_size, _ = timestep.observation.shape
        return jax.random.choice(self.rng, 2, [batch_size, 1]), None
