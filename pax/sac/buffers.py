# modified from https://github.com/henry-prior/jax-rl/blob/master/jax_rl/SAC.py
import jax
import numpy as onp
from haiku import PRNGSequence
from jax import random


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        max_size: int = int(2e6),
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = onp.empty((max_size, state_dim))
        self.action = onp.empty((max_size, 1))
        self.next_state = onp.empty((max_size, state_dim))
        self.reward = onp.empty((max_size, 1))
        self.not_done = onp.empty((max_size, 1))

    def reset(self):
        self.ptr = 0
        self.size = 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, rng: PRNGSequence, batch_size: int):
        ind = random.randint(rng, (batch_size,), 0, self.size)
        return (
            jax.device_put(self.state[ind, :]),
            jax.device_put(self.action[ind, :]),
            jax.device_put(self.next_state[ind, :]),
            jax.device_put(self.reward[ind, :]),
            jax.device_put(self.not_done[ind, :]),
        )

    def add_batch(self, state, action, next_state, reward, done):
        batch_size, state_dim = state.shape
        batch_size_1, action_dim = action.shape

        assert batch_size == batch_size_1
        assert self.state.shape[1] == state_dim

        q, new_ptr = divmod(self.ptr + batch_size, self.max_size)

        if q:
            self.ptr = 0

        new_ptr = min(self.ptr + batch_size, self.max_size)

        self.state[self.ptr : new_ptr] = state
        self.action[self.ptr : new_ptr] = action
        self.next_state[self.ptr : new_ptr] = next_state
        self.reward[self.ptr : new_ptr] = reward.reshape(-1, 1)
        self.not_done[self.ptr : new_ptr] = 1.0 - done.reshape(-1, 1)

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
