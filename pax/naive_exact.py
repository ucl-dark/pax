from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
from dm_env import TimeStep

from pax.env_inner import InfiniteMatrixGame


class TrainingState(NamedTuple):
    # Training state consists of network parameters, random key, timesteps
    params: jnp.ndarray
    timesteps: int
    num_episodes: int
    loss: float


class MemoryState(NamedTuple):
    extras: Mapping[str, jnp.ndarray]
    hidden: None


class Logger:
    metrics: dict


class NaiveExact:
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

        grad_fn = jax.jit(jax.vmap(jax.value_and_grad(_loss, has_aux=True)))

        def policy(state, obs, mem):
            action = state.params
            other_action = obs[:, 5:]

            (loss, _), grad = grad_fn(action, other_action)

            # gradient ascent
            action = action + jnp.multiply(grad, lr)

            # update state
            _state = TrainingState(
                params=action,
                timesteps=state.timesteps + 1,
                num_episodes=state.num_episodes,
                loss=loss.sum(),
            )
            return action, _state, mem

        self._policy = policy

        self.player_id = player_id
        self.action_dim = action_dim
        self.eval = False
        self.num_agents = env.num_envs

        # Set up counters and logger
        self._logger = Logger()
        self._total_steps = 0
        self._logger.metrics = {
            "total_steps": 0,
            "sgd_steps": 0,
            "num_episodes": 0,
            "loss_total": 0,
        }

        self._state = TrainingState(
            jnp.zeros((env.num_envs, 5)),
            timesteps=0,
            num_episodes=0,
            loss=0,
        )

        self._mem = MemoryState({"log_probs": None, "values": None}, 0)

    def make_initial_state(self, t: TimeStep):
        return TrainingState(
            t.observation[:, :5],
            timesteps=0,
            num_episodes=0,
            loss=0,
        ), MemoryState(extras={"log_probs": None, "values": None}, hidden=None)

    def reset_memory(self, *args):
        return self._state

    def select_action(
        self,
        t: TimeStep,
    ) -> jnp.ndarray:

        action, self._state, self._mem = self._policy(
            self._state, t.observation, self._mem
        )
        self._logger.metrics["sgd_steps"] += 1
        self._logger.metrics["loss_total"] = self._state.loss
        self._total_steps += 1
        self._logger.metrics["total_steps"] = self._total_steps
        return action

    def update(self, traj_batch, t_prime, state, mem) -> TrainingState:
        return state, mem
