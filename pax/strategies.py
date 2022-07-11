from functools import partial
from typing import Any, NamedTuple, Tuple
from chex import PRNGKey
import wandb

import jax.numpy as jnp
import jax.random
from dm_env import TimeStep

from pax.env import InfiniteMatrixGame

# states are [CC, DC, CD, DD, START]
# actions are cooperate = 0 or defect = 1


# TODO: Add strategy. PPO should learn to ALL-C
# class ZDExtortion:
#     # @partial(jax.jit, static_argnums=(0,))
#     def __init__(self, *args):
#         self.key = jax.random.PRNGKey(args[0])

#     def select_action(
#         self,
#         timestep: TimeStep,
#     ) -> jnp.ndarray:
#         action, self.key = self._extortion(timestep.observation, self.key)
#         return action

#     def update(self, *args) -> None:
#         pass

#     # @jax.jit
#     def _extortion(self, obs: jnp.ndarray, key):
#         # from https://www.pnas.org/doi/epdf/10.1073/pnas.1206569109
#         # TODO: Remove hard coded cooperation. Make it related to the specific payoff of the game
#         # TODO: What is probability of cooperating in START? Default to 1
#         # CC, CD, DC, DD, START
#         key, subkey = jax.random.split(self.key)
#         samples = jax.random.uniform(
#             subkey, shape=(obs.shape[0],), minval=0.0, maxval=1.0
#         )
#         # TODO: Move this outside the function
#         p_coop = jnp.array(
#             [11 / 13, 1 / 2, 7 / 26, 0, 1]
#         )

#         obs = obs.argmax(axis=-1)
#         action = jnp.where(samples < p_coop[obs], 0, 1)
#         return action, key


class GrimTrigger:
    @partial(jax.jit, static_argnums=(0,))
    def __init__(self, *args):
        pass

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        return self._trigger(timestep.observation)

    def update(self, *args) -> None:
        pass

    def _trigger(self, obs: jnp.ndarray, *args) -> jnp.ndarray:
        # batch_size, _ = obs.shape
        obs = obs.argmax(axis=-1)
        # if 0 | 4 -> C
        # if 1 | 2 | 3 | -> D
        obs = obs % 4  # now either 0, 1, 2, 3
        action = jnp.where(obs > 0.0, 1.0, 0.0)
        return action


class TitForTat:
    @partial(jax.jit, static_argnums=(0,))
    def __init__(self, *args):
        pass

    def select_action(
        self,
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
        # action = jnp.expand_dims(action, axis=-1) # removing this in preference for (action, )
        return action


class Defect:
    @partial(jax.jit, static_argnums=(0,))
    def __init__(self, *args):
        pass

    def select_action(
        self,
        timestep: TimeStep,
    ) -> jnp.ndarray:
        # state is [batch x state_space]
        # return [batch]
        batch_size, _ = timestep.observation.shape
        # return jnp.ones((batch_size, 1))
        return jnp.ones((batch_size,))

    def update(self, *args) -> None:
        pass


class Altruistic:
    @partial(jax.jit, static_argnums=(0,))
    def __init__(self, *args):
        pass

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

    def update(self, *args) -> None:
        pass


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
    @partial(jax.jit, static_argnums=(0,))
    def __init__(self, *args):
        pass

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
        return jnp.ones((batch_size, 5))

    def update(self, *args) -> None:
        pass


class HyperDefect:
    def __init__(self, *args):
        pass

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
        # return jnp.zeros((batch_size, 1))
        return jnp.zeros((batch_size, 5))

    def update(self, *args) -> None:
        pass


class HyperTFT:
    def __init__(self, *args):
        pass

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
        # return jnp.zeros((batch_size, 1))
        return jnp.tile(jnp.array([[1, 0, 1, 0, 1]]), (batch_size, 1))

    def update(self, *args) -> None:
        pass


class TrainingState(NamedTuple):
    # Training state consists of network parameters, random key, timesteps
    params: jnp.ndarray
    random_key: jnp.ndarray
    timesteps: int
    env: InfiniteMatrixGame


class Logger:
    metrics: dict


class NaiveLearner:
    """A Batch of Naive Learners which backprops through the game and updates every step"""

    def __init__(
        self, action_dim: int, env: InfiniteMatrixGame, lr: float, seed: int
    ):

        # Initialise training state (parameters, optimiser state, extras).
        def make_initial_state(
            key: Any, action_dim: int, env: InfiniteMatrixGame
        ) -> TrainingState:
            key, _ = jax.random.split(key)
            params = jax.random.uniform(key, (env.num_envs, action_dim))
            return TrainingState(params, key, timesteps=0, env=env)

        self.lr = lr
        self._state = make_initial_state(seed, action_dim, env)
        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._until_sgd = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
        }

    def select_action(
        self,
        t: TimeStep,
    ) -> jnp.ndarray:
        return self._state.params

    def update(
        self, t: TimeStep, action: jnp.array, t_prime: TimeStep
    ) -> None:
        env = self._state.env
        other_action = t_prime.observation[:, 5:]

        def loss_fn(theta1, theta2, env):
            env.reset()
            # can you just use env._step but debugging atm.
            return env.step([theta1, theta2])[0].reward.sum()

        loss, grad = jax.value_and_grad(loss_fn)(action, other_action, env)
        # gradient ascent
        delta = jnp.multiply(grad, self.lr)
        new_params = self._state.params + delta

        # update state
        self._state = TrainingState(
            params=new_params,
            random_key=self._state.random_key,
            timesteps=self._state.timesteps + 1,
            env=env,
        )

        self._total_steps += 1
        self._logger.metrics["sgd_steps"] += 1
        self._logger.metrics["loss_total"] = loss
        return
