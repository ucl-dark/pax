from typing import Optional

import distrax
import haiku as hk
import jax.numpy as jnp


class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            b_init=hk.initializers.Constant(0),
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Categorical(logits=logits), value)


def make_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                hk.nets.MLP(
                    [64, 64],
                    w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                    b_init=hk.initializers.Constant(0),
                    activate_final=True,
                ),
                CategoricalValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network
