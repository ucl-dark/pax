from dm_env import TimeStep
from typing import Callable, List, Tuple
import jax.numpy as jnp


class IndependentLearners:
    "Interface for a set of batched agents to work with environment"

    def __init__(self, agents: list):
        self.num_agents: int = len(agents)
        self.agents: list = agents

    def select_action(self, timesteps: List[TimeStep]) -> List[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        return [
            agent.select_action(t) for agent, t in zip(self.agents, timesteps)
        ]

    def update(
        self,
        old_timesteps: List[TimeStep],
        actions: List[jnp.ndarray],
        timesteps: List[TimeStep],
    ) -> None:
        # might have to add some centralised training to this
        for agent, t, action, t_1 in zip(
            self.agents, old_timesteps, actions, timesteps
        ):
            agent.update(t, action, t_1)

    def log(self, metrics: List[Callable]) -> None:
        for metric, agent in zip(metrics, self.agents):
            metric(agent)
