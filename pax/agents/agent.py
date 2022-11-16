from typing import Tuple

from pax.utils import MemoryState, TrainingState
import jax.numpy as jnp


class AgentInterface:
    """Interface for agents to interact with runners and environemnts."""

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
