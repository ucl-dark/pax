from typing import Dict, Any, Callable, NamedTuple, Sequence, Tuple, Optional
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
        self._logit_layer = hk.Linear(num_values)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Categorical(logits=logits), value)


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
            # TODO: Remove hardcoded action space.
            CategoricalValueHead(num_values=2),
        ]
    )
    policy_value_network = hk.Sequential(layers)
    return policy_value_network(inputs)


# def critic_network(inputs: jnp.ndarray) -> jnp.ndarray:
#     flat_inputs = hk.Flatten()(inputs)
#     mlp = hk.nets.MLP(
#         [2, 2, 1],
#         w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
#         b_init=hk.initializers.Constant(0),
#         activation=jax.nn.hard_tanh,
#         name="critic_network",
#     )
#     # TODO: Figure out how to properly change the last layer to init with different weights
#     # and to preserve the name (currently it's not prepending critic_network/~/)
#     mlp.layers = mlp.layers[:-1] + (
#         hk.Linear(
#             1,
#             w_init=hk.initializers.Orthogonal(1.0),
#             b_init=hk.initializers.Constant(0),
#             name="linear_%d" % (len(mlp.layers) - 1),
#         ),
#     )
#     policy = mlp(flat_inputs)
#     return policy


# def actor_network(inputs: jnp.ndarray) -> jnp.ndarray:
#     flat_inputs = hk.Flatten()(inputs)
#     mlp = hk.nets.MLP(
#         [2, 2, 2],
#         w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
#         b_init=hk.initializers.Constant(0),
#         activation=jax.nn.hard_tanh,
#         name="actor_network",
#     )
#     # TODO: Figure out how to properly change the last layer to init with different weights
#     # and to preserve the name (currently it's not prepending actor_network/~/)
#     mlp.layers = mlp.layers[:-1] + (
#         hk.Linear(
#             2,
#             w_init=hk.initializers.Orthogonal(0.01),
#             b_init=hk.initializers.Constant(0),
#             name="linear_%d" % (len(mlp.layers) - 1),
#         ),
#     )
#     actions = mlp(flat_inputs)
#     return actions
