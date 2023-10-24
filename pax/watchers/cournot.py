from functools import partial

import jax
from jax import numpy as jnp

from pax.envs.cournot import EnvParams as CournotEnvParams, CournotGame


@partial(jax.jit, static_argnums=2)
def cournot_stats(
    observations: jnp.ndarray, params: CournotEnvParams, num_players: int
) -> dict:
    opt_quantity = CournotGame.nash_policy(params)

    actions = observations[..., :num_players]
    average_quantity = actions.mean()

    stats = {
        "cournot/average_quantity": average_quantity,
        "cournot/quantity_loss": jnp.mean(
            (opt_quantity - average_quantity) ** 2
        ),
        "cournot/price": observations[..., -1],
    }

    for i in range(num_players):
        stats["cournot/quantity_" + str(i)] = jnp.mean(observations[..., i])

    return stats
