from typing import Callable, List

import jax.numpy as jnp
from dm_env import TimeStep
import jax


class IndependentLearners:
    "Interface for a set of batched agents to work with environment"

    def __init__(self, agents: list, args: dict):
        self.num_agents: int = len(agents)
        self.agents: list = agents
        self.args = args

        # vmap agents to work in respective position
        agent1, agent2 = self.agents

        # batch MemoryState not TrainingState
        agent1.batch_init = jax.vmap(
            agent1.make_initial_state,
            (None, 0),
            (None, 0),
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(agent1.reset_memory, (0, None), 0), static_argnums=1
        )
        agent1.batch_policy = jax.jit(
            jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0))
        )

        # batch all for Agent2
        if self.args.agent2 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy, 0, 0))
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

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

    def eval(self, set_flag: bool) -> None:
        for agent in self.agents:
            agent.eval = set_flag
