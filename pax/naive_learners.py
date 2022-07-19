from typing import NamedTuple
import jax.numpy as jnp

from pax.env import InfiniteMatrixGame
from dm_env import TimeStep


class TrainingState(NamedTuple):
    # Training state consists of network parameters, random key, timesteps
    params: jnp.ndarray
    timesteps: int
    num_episodes: int
    env: InfiniteMatrixGame


class Logger:
    metrics: dict


class NaiveLearnerEx:
    """A Batch of Naive Learners which backprops through the game and updates every step"""

    def __init__(
        self,
        action_dim: int,
        env: InfiniteMatrixGame,
        lr: float,
        seed: int,
        player_id: int,
    ):

        # Initialise training state (parameters, optimiser state, extras).

        self.lr = lr
        self.player_id = player_id
        self.action_dim = action_dim
        self.eval = False
        self.num_agents = env.num_envs
        self._state = TrainingState(0, timesteps=0, num_episodes=0, env=env)
        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "num_episodes": 0,
            "loss_total": 0,
        }

    def select_action(
        self,
        t: TimeStep,
    ) -> jnp.ndarray:

        if t.first():
            # use the random init provided by env (and seen by other agent)
            action = t.observation[:, :5]
            self._logger.metrics["num_episodes"] += self._state.num_episodes

        else:
            action = self._state.params
        other_action = t.observation[:, 5:]
        (loss, _), grad = self._state.env.grad(action, other_action)

        # gradient ascent
        new_params = action + jnp.multiply(grad, self.lr)

        # update state
        self._state = TrainingState(
            params=new_params,
            timesteps=self._state.timesteps + 1,
            num_episodes=self._state.num_episodes,
            env=self._state.env,
        )

        self._logger.metrics["sgd_steps"] += 1
        self._logger.metrics["loss_total"] = loss.mean(axis=0)
        self._total_steps += 1
        self._logger.metrics["total_steps"] = self._total_steps
        return self._state.params

    def update(
        self, t: TimeStep, action: jnp.array, t_prime: TimeStep
    ) -> None:
        pass
