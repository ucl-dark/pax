from typing import Tuple

import jax.numpy as jnp

from pax.utils import MemoryState, TrainingState


class AgentInterface:
    """Interface for agents to interact with runners and environemnts.
    Note this is not actually enforce by Python (we don't use ABCs) but
    it's a useful convention to follow.
    """

    def make_initial_state(self, *args) -> Tuple[TrainingState, MemoryState]:
        """Make an initial state for the agent to start from."""
        raise NotImplementedError

    def reset_memory(self, *args) -> MemoryState:
        """Reset the memory state of the agent."""
        raise NotImplementedError

    def policy(self, *args) -> Tuple[jnp.array, TrainingState, MemoryState]:
        """Make a policy decision."""
        raise NotImplementedError

    def update(self, *args) -> Tuple[TrainingState, MemoryState]:
        """Update the agent's state."""
        raise NotImplementedError
