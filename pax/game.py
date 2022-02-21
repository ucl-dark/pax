import enum
from functools import partial
from turtle import done
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from dm_env import Environment, TimeStep, restart, specs, termination, transition

import wandb
from src.strategies import Altruistic, Defect, Random, TitForTat

# payoffs are [(2, 2), (3,0), (0,3), (1, 1)]
# observations are one-hot representations


class State(enum.IntEnum):
    CC = 0
    DC = 1
    CD = 2
    DD = 3
    START = 4


_ACTIONS = (0, 1)  # Cooperate, Defect
_STATES = (0, 1, 2, 3, 4)  # CC, DC, CD, DD, START


class IteratedPrisonersDilemma(Environment):
    def __init__(self, episode_length: int, num_envs: int) -> None:
        self.episode_length = episode_length
        self.num_envs = num_envs
        self.n_agents = 2

        self._num_steps = 0
        self._reset_next_step = True

    def step(
        self, action_1: jnp.array, action_2: jnp.ndarray
    ) -> Tuple[TimeStep, TimeStep]:
        if self._reset_next_step:
            return self.reset()

        self._num_steps += 1
        assert action_1.shape == action_2.shape
        assert action_1.shape == (self.num_envs, 1)

        r_1, r_2 = self._get_reward(action_1, action_2)
        state = self._get_state(action_1, action_2)
        obs_1, obs_2 = self._observation(state)

        if self._num_steps == self.episode_length:
            self._reset_next_step = True
            return termination(reward=r_1, observation=obs_1), termination(
                reward=r_2, observation=obs_2
            )

        return transition(reward=r_1, observation=obs_1), transition(
            reward=r_2, observation=obs_2
        )

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(num_values=len(State), name="previous turn")

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(dtype=int, num_values=len(_ACTIONS), name="action")

    def reset(self) -> Tuple[TimeStep, TimeStep]:
        """Returns the first `TimeStep` of a new episode."""
        self._reset_next_step = False
        obs_1, obs_2 = self._observation(State.START * jnp.ones((self.num_envs, 1)))
        self._num_steps = 0
        return restart(obs_1), restart(obs_2)

    @partial(jax.jit, static_argnums=(0,))
    def _get_reward(self, a1, a2) -> Tuple[jnp.array, jnp.array]:
        cc = 2 * (a1 - 1) * (a2 - 1)
        dd = a1 * a2
        dc = 3 * a1 * (1 - a2)
        cd = 3 * (1 - a1) * a2
        r1 = cc + dd + dc
        r2 = cc + dd + cd
        return r1, r2

    @partial(jax.jit, static_argnums=(0,))
    def _get_state(self, a1: jnp.array, a2: jnp.array) -> jnp.array:
        return (
            0 * (1 - a1) * (1 - a2)
            + 1 * (a1) * (1 - a2)
            + 2 * (1 - a1) * (a2)
            + 3 * (a1) * (a2)
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def _observation(self, state: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # each agent wants to assume they are agent 1 canonicaly:
        # observation also changes environment to be state space
        o_state = jnp.array(state, copy=True)

        # switch the assymetric states (CD and DC)
        # 0 -> 0, 1 -> 2, 2 -> 1, 3 -> 3
        s0 = state
        s1 = State.DC - state
        s2 = State.CD - state
        s3 = State.DD - state
        s4 = State.START - state

        # coefficients are calculated to guarantee switch we want
        state = (
            0 * s1 * s2 * s3 * s4
            + (2 / (1 * 2 * 3)) * s0 * s2 * s3 * s4
            + (1 / (2 * (-1) * 1 * 2)) * s0 * s1 * s3 * s4
            + (3 / (3 * (-2) * (-1) * 1)) * s0 * s1 * s2 * s4
            + (4 / (4 * (-3) * (-2) * (-1))) * s0 * s1 * s2 * s3
        )

        # make this one hot
        return jax.nn.one_hot(o_state, len(State)).squeeze(1), jax.nn.one_hot(
            state, len(State)
        ).squeeze(1)


def evaluate(
    agent_0, agent_1, seed: int, episode_length: int, num_episodes: int, logging: bool
):
    num_envs = 1
    key_0 = jax.random.PRNGKey(seed)
    key_1 = jax.random.PRNGKey(seed + 1)
    env = IteratedPrisonersDilemma(
        episode_length,
        num_envs,
    )

    print("Starting Prisoners Dilemma")
    print(f"Max time limit {episode_length}")
    print("-----------------------")

    # batch_size x action_space
    rewards_0 = []
    rewards_1 = []
    for _ in range(num_episodes):
        _timestep_0, _timestep_1 = env.reset()
        for t in range(episode_length):
            _action_0, _ = agent_0.actor_step(key_0, _timestep_0.observation, True)
            _action_1, _ = agent_1.actor_step(key_1, _timestep_1.observation, True)

            if not logging:
                print(
                    f"player 1 observes: {_timestep_0.observation}, player 2 observes: {_timestep_1.observation}"
                )
                print(f"player 1 action: {_action_0}, player 2 action: {_action_1}")

            _timestep_0, _timestep_1 = env.step(_action_0, _action_1)

            if logging:
                wandb.log(
                    {
                        f"Test Reward Player 1: {type(agent_0).__name__} vs {type(agent_1).__name__}": _r1,
                        "Time Step": t,
                    }
                )
            else:
                print(f"Rewards: {_timestep_0.reward}, {_timestep_1.reward}")
                print(f"Done: {_timestep_0.last()},  {_timestep_1.last()}")

            rewards_0.append(_timestep_0.reward)
            rewards_1.append(_timestep_1.reward)
            # next episode
            _, key_0 = jax.random.split(key_0)
            _, key_1 = jax.random.split(key_0)

    rewards_0 = jnp.asarray(rewards_0)
    rewards_1 = jnp.asarray(rewards_1)

    if logging:
        wandb.log(
            {
                f"Mean Reward Player 1: {type(agent_0).__name__} vs {type(agent_1).__name__}": float(
                    rewards_0[:, :].mean()
                ),
                f"Mean Reward Player 2: {type(agent_0).__name__} vs {type(agent_1).__name__}": float(
                    rewards_1[:, :, 1].mean()
                ),
            }
        )
    else:
        print(
            f"Total Evaluation Rewards: {float(rewards_0.mean()),float(rewards_1.mean())}"
        )

    return rewards_1


if __name__ == "__main__":
    agent_0 = Altruistic()
    agent_1 = TitForTat()
    evaluate(agent_0, agent_1, 0, 50, 1, False)
