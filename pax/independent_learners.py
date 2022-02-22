from dm_env import TimeStep
from typing import List, Tuple
import jax.numpy as jnp


class IndependentLeaners:
    "Interface for a set of agents to work with environment"

    def __init__(self, list_of_agents: list):
        self.num_agents = len(list_of_agents)
        self.agents = list_of_agents

    def select_action(self, timesteps: List[TimeStep]) -> Tuple[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        return (agent.select_action(t) for agent, t in zip(self.agents, timesteps))
