import haiku as hk
import jax.numpy as jnp
import jax


def critic_network(inputs: jnp.ndarray) -> jnp.ndarray:
    flat_inputs = hk.Flatten()(inputs)
    mlp = hk.nets.MLP(
        [2, 2, 1],
        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
        b_init=hk.initializers.Constant(0),
        activation=jax.nn.hard_tanh,
        name="critic_network",
    )
    # TODO: Figure out how to properly change the last layer to init with different weights
    # and to preserve the name (currently it's not prepending critic_network/~/)
    mlp.layers = mlp.layers[:-1] + (
        hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            b_init=hk.initializers.Constant(0),
            name="linear_%d" % (len(mlp.layers) - 1),
        ),
    )
    policy = mlp(flat_inputs)
    return policy


def actor_network(inputs: jnp.ndarray) -> jnp.ndarray:
    flat_inputs = hk.Flatten()(inputs)
    mlp = hk.nets.MLP(
        [2, 2, 2],
        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
        b_init=hk.initializers.Constant(0),
        activation=jax.nn.hard_tanh,
        name="actor_network",
    )
    # TODO: Figure out how to properly change the last layer to init with different weights
    # and to preserve the name (currently it's not prepending actor_network/~/)
    mlp.layers = mlp.layers[:-1] + (
        hk.Linear(
            2,
            w_init=hk.initializers.Orthogonal(0.01),
            b_init=hk.initializers.Constant(0),
            name="linear_%d" % (len(mlp.layers) - 1),
        ),
    )
    actions = mlp(flat_inputs)
    return actions
