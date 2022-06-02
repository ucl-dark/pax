from dm_env import TimeStep
from typing import Callable, List
import jax.numpy as jnp
import jax


class IndependentLearners:
    "Interface for a set of batched agents to work with environment"

    def __init__(self, agents: list):
        self.num_agents: int = len(agents)
        self.agents: list = agents

    def select_action(
        self, key, timesteps: List[TimeStep]
    ) -> List[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        keys = jax.random.split(key, num=self.num_agents)
        return [
            agent.select_action(prngkey, t)
            for agent, prngkey, t in zip(self.agents, keys, timesteps)
        ]

    def update(
        self,
        old_timesteps: List[TimeStep],
        actions: List[jnp.ndarray],
        timesteps: List[TimeStep],
        key,
    ) -> None:
        # might have to add some centralised training to this
        for agent, t, action, t_1 in zip(
            self.agents, old_timesteps, actions, timesteps
        ):
            agent.update(t, action, t_1, key)

    def log(self, metrics: List[Callable]) -> None:
        for metric, agent in zip(metrics, self.agents):
            metric(agent)

    def eval(self, set_flag: bool) -> None:
        for agent in self.agents:
            agent.eval = set_flag
