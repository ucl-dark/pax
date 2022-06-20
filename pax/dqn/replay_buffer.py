# adapted from https://github.com/deepmind/bsuite/blob/master/bsuite/baselines/utils/replay.py
"""A simple, uniformly sampled replay buffer with batch adding."""

from typing import Any, Optional, Sequence

import numpy as np
import jax


class Replay:
    """Uniform replay buffer. Allocates all required memory at initialization."""

    _data: Optional[Sequence[np.ndarray]]
    _capacity: int
    _num_added: int

    def __init__(self, capacity: int):
        """Initializes a new `Replay`.
        Args:
        capacity: The maximum number of items allowed in the replay. Adding
            items to a replay that is at maximum capacity will overwrite the oldest
            items.
        """
        self._data = None
        self._capacity = capacity
        self._num_added = 0
        # added
        self.ptr = 0

    def add(self, items: Sequence[Any]):
        """Adds a single sequence of items to the replay.
        Args:
        items: Sequence of items to add. Does not handle batched or nested items.
        """
        if self._data is None:
            self._preallocate(items)

        for slot, item in zip(self._data, items):
            slot[self._num_added % self._capacity] = item

        self._num_added += 1

    def add_batch(self, items: Sequence[Any], batch_size):
        """Adds a batched sequence of items to the replay.
        Args:
        items: Sequence of items to add. Handles batched items.
        """

        if self._data is None:
            self._preallocate_batched(items)

        q, new_ptr = divmod(self.ptr + batch_size, self._capacity)

        if q:
            self.ptr = 0

        new_ptr = min(self.ptr + batch_size, self._capacity)

        for slot, item in zip(self._data, items):
            slot[self.ptr : new_ptr] = item

        self.ptr = (self.ptr + batch_size) % self._capacity
        self._num_added += batch_size

    def sample(self, size: int, key) -> Sequence[np.ndarray]:
        """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
        # numpy implementation w/o jax
        # indices = np.random.randint(self.size, size=size)
        key, subkey = jax.random.split(key)
        indices = jax.random.randint(
            subkey, shape=(size,), minval=0, maxval=self.size
        )
        return [slot[indices] for slot in self._data], key

    def reset(
        self,
    ):
        """Resets the replay."""
        self._data = None

    @property
    def size(self) -> int:
        return min(self._capacity, self._num_added)

    @property
    def fraction_filled(self) -> float:
        return self.size / self._capacity

    def _preallocate(self, items: Sequence[Any]):
        """Assume flat structure of items."""
        as_array = []
        for item in items:
            if item is None:
                raise ValueError("Cannot store `None` objects in replay.")
            as_array.append(np.asarray(item))

        self._data = [
            np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape)
            for x in as_array
        ]

    def _preallocate_batched(self, items: Sequence[Any]):
        """Preallocates batched structures items."""
        as_array = []
        for item in items:
            if item is None:
                raise ValueError("Cannot store `None` objects in replay.")
            as_array.append(np.asarray(item))

        self._data = [
            np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape[1:])
            for x in as_array
        ]

    def __repr__(self):
        return "Replay: size={}, capacity={}, num_added={}".format(
            self.size, self._capacity, self._num_added
        )
