from typing import Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from pax import utils

class DeterministicFunction(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None):
        super().__init__(name=name)
        self._act_body = hk.nets.MLP(
                    [64, 64],
                    w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                    b_init=hk.initializers.Constant(0),
                    activate_final=True,
                    activation=jax.nn.relu,
                    )
        self._act_output = hk.nets.MLP(
                    [num_values],
                    w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                    b_init=hk.initializers.Constant(0),
                    activate_final=True,
                    activation=jnp.tanh,
                    )


    def __call__(self, inputs: jnp.ndarray):
        output = self._act_body(inputs)
        output = self._act_output(output)

        return output


def make_act_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                DeterministicFunction(num_values=num_actions)
            ]
        )
        act_network = hk.Sequential(layers)
        return act_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network