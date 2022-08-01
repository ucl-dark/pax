from typing import Callable, List

from dm_env import TimeStep
import jax.numpy as jnp


class CentralizedLearners:
    """Interface for a set of batched agents to work with environment
    Performs centralized training"""

    def __init__(self, agents: list):
        self.num_agents: int = len(agents)
        self.agents: list = agents

    def select_action(self, timesteps: List[TimeStep]) -> List[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        return [
            agent.select_action(t) for agent, t in zip(self.agents, timesteps)
        ]

    def in_lookahead(self, env):
        """Simulates a rollout and gradient update"""
        counter = 0
        for agent in self.agents:
            # All other agents in a list
            # i.e. if i am agent2, then other_agents=[agent1, agent3, agent4 ...]
            other_agents = self.agents[:counter] + self.agents[counter + 1 :]
            agent.in_lookahead(env, other_agents)
            counter += 1

    def out_lookahead(self, env):
        """Performs a real rollout and update"""
        counter = 0
        for agent in self.agents:
            # All other agents in a list
            # i.e. if i am agent2, then other_agents=[agent1, agent3, agent4 ...]
            other_agents = self.agents[:counter] + self.agents[counter + 1 :]
            agent.out_lookahead(env, other_agents)
            counter += 1

    # TODO: Obselete at the moment. This can be put into the LOLA.
    def update(
        self,
        old_timesteps: List[TimeStep],
        actions: List[jnp.ndarray],
        timesteps: List[TimeStep],
    ) -> None:
        counter = 0
        for agent, t, action, t_1 in zip(
            self.agents, old_timesteps, actions, timesteps
        ):
            # All other agents in a list
            # i.e. if i am agent2, then other_agents=[agent1, agent3, agent4 ...]
            other_agents = self.agents[:counter] + self.agents[counter + 1 :]
            agent.update(t, action, t_1, other_agents)
            counter += 1

    def log(self, metrics: List[Callable]) -> None:
        for metric, agent in zip(metrics, self.agents):
            metric(agent)

    def eval(self, set_flag: bool) -> None:
        for agent in self.agents:
            agent.eval = set_flag
