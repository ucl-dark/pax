from typing import NamedTuple
import jax
import jax.numpy as jnp

from pax.env import InfiniteMatrixGame
from dm_env import TimeStep


class TrainingState(NamedTuple):
    # Training state consists of network parameters, random key, timesteps
    params: jnp.ndarray
    timesteps: int
    num_episodes: int


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
        payout_mat_1 = jnp.array([[r[0] for r in env.payoff]])
        gamma = env.gamma

        def _loss(theta1, theta2):
            theta1 = jax.nn.sigmoid(theta1)
            _th2 = jnp.array(
                [theta2[0], theta2[2], theta2[1], theta2[3], theta2[4]]
            )

            p_1_0 = theta1[4:5]
            p_2_0 = _th2[4:5]
            p = jnp.concatenate(
                [
                    p_1_0 * p_2_0,
                    p_1_0 * (1 - p_2_0),
                    (1 - p_1_0) * p_2_0,
                    (1 - p_1_0) * (1 - p_2_0),
                ]
            )
            p_1 = jnp.reshape(theta1[0:4], (4, 1))
            p_2 = jnp.reshape(_th2[0:4], (4, 1))
            P = jnp.concatenate(
                [
                    p_1 * p_2,
                    p_1 * (1 - p_2),
                    (1 - p_1) * p_2,
                    (1 - p_1) * (1 - p_2),
                ],
                axis=1,
            )
            M = jnp.matmul(p, jnp.linalg.inv(jnp.eye(4) - gamma * P))
            L_1 = jnp.matmul(M, jnp.reshape(payout_mat_1, (4, 1)))
            return L_1.sum(), M

        self.grad = jax.jit(jax.vmap(jax.value_and_grad(_loss, has_aux=True)))
        self.lr = lr
        self.player_id = player_id
        self.action_dim = action_dim
        self.eval = False
        self.num_agents = env.num_envs

        self._state = TrainingState(
            jnp.zeros((self.num_agents, 5)), timesteps=0, num_episodes=0
        )

        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "num_episodes": 0,
            "loss_total": 0,
        }

    def reset_state(self, t: TimeStep):
        return TrainingState(t.observation[:, :5], timesteps=0, num_episodes=0)

    def select_action(
        self,
        t: TimeStep,
    ) -> jnp.ndarray:

        action = self._state.params
        other_action = t[:, 5:]
        (loss, _), grad = self.grad(action, other_action)

        # gradient ascent
        new_params = action + jnp.multiply(grad, self.lr)

        # update state
        self._state = TrainingState(
            params=new_params,
            timesteps=self._state.timesteps + 1,
            num_episodes=self._state.num_episodes,
        )

        self._logger.metrics["sgd_steps"] += 1
        self._logger.metrics["loss_total"] = loss.mean(axis=0)
        self._total_steps += 1
        self._logger.metrics["total_steps"] = self._total_steps
        return self._state.params

    def _policy(self, params, obs, state):
        action = params
        other_action = obs[:, 5:]

        (loss, _), grad = self.grad(action, other_action)

        # gradient ascent
        new_params = action + jnp.multiply(grad, self.lr)

        # update state
        _state = TrainingState(
            params=new_params,
            timesteps=state.timesteps + 1,
            num_episodes=state.num_episodes,
        )
        return new_params, _state

    def update(
        self, t: TimeStep, action: jnp.array, t_prime: TimeStep
    ) -> None:
        pass
