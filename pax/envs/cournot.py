from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces


@chex.dataclass
class EnvState:
    inner_t: int
    outer_t: int


@chex.dataclass
class EnvParams:
    a: int
    b: float
    marginal_cost: float


class CournotGame(environment.Environment):
    def __init__(self, num_players: int, num_inner_steps: int):
        super().__init__()
        self.num_players = num_players

        def _step(
            key: chex.PRNGKey,
            state: EnvState,
            actions: Tuple[float, ...],
            params: EnvParams,
        ):
            assert len(actions) == num_players
            t = state.outer_t
            done = t >= num_inner_steps
            key, _ = jax.random.split(key, 2)

            actions = jnp.asarray(actions).squeeze()
            actions = jnp.clip(actions, a_min=0)
            p = params.a - params.b * actions.sum()

            all_obs = []
            all_rewards = []
            for i in range(num_players):
                q = actions[i]
                obs = jnp.concatenate([actions, jnp.array([p])])
                all_obs.append(obs)

                r = p * q - params.marginal_cost * q
                all_rewards.append(r)

            state = EnvState(
                inner_t=state.inner_t + 1, outer_t=state.outer_t + 1
            )

            return (
                tuple(all_obs),
                state,
                tuple(all_rewards),
                done,
                {},
            )

        def _reset(
            key: chex.PRNGKey, params: EnvParams
        ) -> Tuple[Tuple, EnvState]:
            state = EnvState(
                inner_t=jnp.zeros((), dtype=jnp.int8),
                outer_t=jnp.zeros((), dtype=jnp.int8),
            )
            obs = jax.random.uniform(key, (num_players + 1,))
            obs = jnp.concatenate([obs])
            return tuple([obs for _ in range(num_players)]), state

        self.step = jax.jit(_step)
        self.reset = jax.jit(_reset)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Cournot-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=0, high=float("inf"), shape=(1,))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=0,
            high=float("inf"),
            shape=self.num_players + 1,
            dtype=jnp.float32,
        )

    @staticmethod
    def nash_policy(params: EnvParams) -> float:
        return 2 * (params.a - params.marginal_cost) / (3 * params.b)

    @staticmethod
    def nash_reward(params: EnvParams) -> float:
        q = CournotGame.nash_policy(params)
        p = params.a - params.b * q
        return p * q - params.marginal_cost * q
