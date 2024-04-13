from tokenize import String

import gym
import jax.numpy as jnp
from dm_env import Environment, TimeStep, specs

from . import gym_wrapper


class BatchedEnvs(Environment):
    """Batches a number of cartpole environments that are called sequentially.
    Similar to OpenAIs SyncVectorEnv"""

    def __init__(
        self, num_envs: int, seed: int = 0, env_id: String = "CartPole-v1"
    ):
        def make_env(env_id, seed):
            """Creates a gym environment and converts it to an dm_env"""
            env = gym.make(env_id)
            env.reset(seed=seed)
            env = gym_wrapper.DMEnvFromGym(env)
            return env

        self.envs = [
            make_env(env_id, seed + i) for i in range(1, num_envs + 1)
        ]

    def step(self, actions: jnp.ndarray) -> TimeStep:
        """Step multiple environments"""
        step_types = []
        rewards = []
        observations = []
        discounts = []
        for env, action in zip(self.envs, actions):
            t = env.step(int(action))
            if t.step_type == 2:
                t_reset = env.reset()
                observations.append(t_reset.observation)
            else:
                observations.append(t.observation)
            step_types.append(t.step_type)
            rewards.append(t.reward)
            discounts.append(t.discount)
        timestep = TimeStep(
            step_type=jnp.array(step_types),
            observation=jnp.array(observations),
            reward=jnp.array(rewards),
            discount=jnp.array(discounts),
        )
        return timestep

    def reset(self):
        """Resets multiple environments"""
        observations = []
        rewards = []
        discounts = []
        step_types = []
        for env in self.envs:
            t = env.reset()
            step_types.append(t.step_type)
            observations.append(t.observation)
            rewards.append(t.reward)
            discounts.append(t.discount)
        timestep = TimeStep(
            step_type=jnp.array(step_types),
            observation=jnp.array(observations),
            reward=jnp.array(rewards),
            discount=jnp.array(discounts),
        )
        return timestep

    def observation_spec(self) -> specs.DiscreteArray:
        """Returns the observation spec."""
        return specs.DiscreteArray(
            num_values=self.envs[0].observation_spec().shape[0],
            name="previous turn",
        )

    def action_spec(self) -> specs.DiscreteArray:
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int,
            num_values=self.envs[0].action_spec().num_values,
            name="action",
        )


if __name__ == "__main__":
    pass
