from tokenize import String
from dm_env import (
    Environment,
    TimeStep,
    restart,
    specs,
    termination,
    transition,
    StepType,
)
import gym
import jax.numpy as jnp
from bsuite.utils import gym_wrapper


class BatchedEnvs(Environment):
    """Batches a number of cartpole environments that are called sequentially.
    Similar to OpenAIs SyncVectorEnv"""

    # TODO: Extend to all gym environments

    def __init__(self, num_envs: int, seed: int, env_id: String):
        seed = 0
        env_id = "CartPole-v1"

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
        observations = []
        rewards = []
        discounts = []
        step_types = []
        for env, action in zip(self.envs, actions):
            t = env.step(int(action))
            if t.step_type == 2:
                step_types.append(1)
            else:
                step_types.append(0)
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

    def reset(self):
        """Resets multiple environments"""
        observations = []
        rewards = []
        discounts = []
        step_types = []
        for env in self.envs:
            t = env.reset()
            if t.step_type == 2:
                step_types.append(1)
            else:
                step_types.append(0)
            # step_types.append(StepType(t.step_type))
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
    seed = 0
    env_id = "CartPole-v1"
    num_envs = 2
    env = BatchedEnvs(num_envs=num_envs, seed=seed, env_id=env_id)
    print(env.observation_spec().num_values)
    t = env.reset()
    print(t)
    print(t.observation.shape)
    actions = jnp.array([0, 0])
    while True:
        t_prime = env.step(actions)
        print(t_prime)
        if t_prime.step_type[0] == True:
            break
