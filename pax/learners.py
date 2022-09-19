from typing import Callable, List

import jax
import jax.numpy as jnp
from dm_env import TimeStep


class IndependentLearners:
    "Interface for a set of batched agents to work with environment"

    def __init__(self, agents: list, args: dict):
        self.num_agents: int = len(agents)
        self.agents: list = agents
        self.args = args

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
        if args.agent2 == "NaiveEx":
            # special case where NaiveEx has a different call signature
            agent2.batch_init = jax.jit(jax.vmap(agent2.make_initial_state))
        else:
            agent2.batch_init = jax.vmap(
                agent2.make_initial_state, (0, None), 0
            )
        agent2.batch_policy = jax.jit(jax.vmap(agent2._policy))
        agent2.batch_reset = jax.jit(
            jax.vmap(agent2.reset_memory, (0, None), 0), static_argnums=1
        )
        agent2.batch_update = jax.jit(jax.vmap(agent2.update, (1, 0, 0, 0), 0))

        # init agents
        init_hidden = jnp.tile(agent1._mem.hidden, (args.num_opps, 1, 1))
        agent1._state, agent1._mem = agent1.batch_init(
            agent1._state.random_key, init_hidden
        )

        if args.agent2 != "NaiveEx":
            # NaiveEx requires env first step to init.
            init_hidden = jnp.tile(agent2._mem.hidden, (args.num_opps, 1, 1))
            agent2._state, agent2._mem = agent2.batch_init(
                jax.random.split(agent2._state.random_key, args.num_opps),
                init_hidden,
            )

    def select_action(self, timesteps: List[TimeStep]) -> List[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        return [
            agent.select_action(t) for agent, t in zip(self.agents, timesteps)
        ]

    def log(self, metrics: List[Callable]) -> None:
        for metric, agent in zip(metrics, self.agents):
            metric(agent)

    def eval(self, set_flag: bool) -> None:
        for agent in self.agents:
            agent.eval = set_flag


class EvolutionaryLearners:
    """
    Interface for a set of batched evolutionary agents to work with environment.
    This is essentially the same as IndependentLeaners with an added
    dimension for population size.
    """

    def __init__(self, agents: list, args: dict):
        self.num_agents: int = len(agents)
        self.agents: list = agents
        self.args = args

        agent1, agent2 = self.agents
        # batch MemoryState not TrainingState
        agent1.batch_init = jax.vmap(
            jax.vmap(
                agent1.make_initial_state,
                (None, 0),
                (None, 0),
            ),
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent1.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent1.batch_policy = jax.jit(
            jax.vmap(
                jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0)),
            )
        )

        agent2.batch_init = jax.vmap(
            jax.vmap(
                agent1.make_initial_state,
                (None, 0),
                (None, 0),
            ),
        )
        agent2.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent1.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent2.batch_policy = jax.jit(
            jax.vmap(
                jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0)),
            )
        )

    def select_action(self, timesteps: List[TimeStep]) -> List[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        return [
            agent.select_action(t) for agent, t in zip(self.agents, timesteps)
        ]

    def log(self, metrics: List[Callable]) -> None:
        for metric, agent in zip(metrics, self.agents):
            metric(agent)

    def eval(self, set_flag: bool) -> None:
        for agent in self.agents:
            agent.eval = set_flag


class EvolutionarySPLearners:
    """
    Interface for a set of batched evolutionary agents to work with environment.
    This is essentially the same as IndependentLeaners with an added
    dimension for population size.
    """

    def __init__(self, agents: list, args: dict):
        self.num_agents: int = len(agents)
        self.agents: list = agents
        self.args = args

        agent1, agent2 = self.agents
        # batch MemoryState not TrainingState
        agent1.batch_init = jax.vmap(
            jax.vmap(
                agent1.make_initial_state,
                (None, 0),
                (None, 0),
            ),
        )
        agent1.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent1.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent1.batch_policy = jax.jit(
            jax.vmap(
                jax.vmap(agent1._policy, (None, 0, 0), (0, None, 0)),
            )
        )

        agent2.batch_init = jax.vmap(
            jax.vmap(
                agent2.make_initial_state,
                (None, 0),
                (None, 0),
            ),
        )
        agent2.batch_reset = jax.jit(
            jax.vmap(
                jax.vmap(agent2.reset_memory, (0, None), 0), (0, None), 0
            ),
            static_argnums=1,
        )

        agent2.batch_policy = jax.jit(
            jax.vmap(
                jax.vmap(agent2._policy, (None, 0, 0), (0, None, 0)),
            )
        )

    def select_action(self, timesteps: List[TimeStep]) -> List[jnp.ndarray]:
        assert len(timesteps) == self.num_agents
        return [
            agent.select_action(t) for agent, t in zip(self.agents, timesteps)
        ]

    def log(self, metrics: List[Callable]) -> None:
        for metric, agent in zip(metrics, self.agents):
            metric(agent)

    def eval(self, set_flag: bool) -> None:
        for agent in self.agents:
            agent.eval = set_flag
