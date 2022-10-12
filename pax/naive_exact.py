from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
from dm_env import TimeStep

from pax.env_inner import InfiniteMatrixGame
from pax.utils import MemoryState


class TrainingState(NamedTuple):
    # Training state consists of network parameters, random key, timesteps
    timesteps: int
    num_episodes: int
    # loss: float


class Logger:
    metrics: dict


class NaiveExact:
    """A Batch of Naive Learners which backprops through the game and updates every step"""

    def __init__(
        self,
        action_dim: int,
        env: InfiniteMatrixGame,
        lr: float,
        num_envs: int,
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

        grad_fn = jax.vmap(jax.value_and_grad(_loss, has_aux=True))

        def policy(state, obs, mem):
            action = mem.hidden
            other_action = obs[..., 5:]
            (loss, _), grad = grad_fn(action, other_action)

            # gradient ascent
            action = action + jnp.multiply(grad, lr)

            # update state
            _state = TrainingState(
                timesteps=state.timesteps + 1,
                num_episodes=state.num_episodes,
            )
            _mem = mem._replace(hidden=action)
            return action, _state, _mem

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
            # jnp.zeros((env.num_envs, 5)),
            timesteps=0,
            num_episodes=0,
            # loss=0,
        )

        self._mem = MemoryState(
            hidden=jnp.zeros((num_envs, 5)),
            extras={
                "values": jnp.zeros(num_envs),
                "log_probs": jnp.zeros(num_envs),
            },
        )

    def make_initial_state(self, t: TimeStep):
        num_envs = t.reward.shape[-1]
        return (
            TrainingState(
                t.observation[..., :5],
                timesteps=0,
                num_episodes=0,
                # loss=0,
            ),
            MemoryState(
                hidden=t.observation[:, :5],
                extras={
                    "values": jnp.zeros(num_envs),
                    "log_probs": jnp.zeros(num_envs),
                },
            ),
        )

    def reset_memory(self, mem, *args):
        return mem

    def select_action(
        self,
        t: TimeStep,
    ) -> jnp.ndarray:

        action, self._state, self._mem = self._policy(
            self._state, t.observation, self._mem
        )
        self._logger.metrics["sgd_steps"] += 1
        # self._logger.metrics["loss_total"] = self._state.loss
        self._total_steps += 1
        self._logger.metrics["total_steps"] = self._total_steps
        return action

    def update(self, traj_batch, t_prime, state, mem) -> TrainingState:
        return state, mem, {}
