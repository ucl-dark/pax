from typing import NamedTuple

from jax import numpy as jnp

from pax.envs.cournot import EnvParams as CournotEnvParams, CournotGame


def cournot_stats(traj1: NamedTuple, traj2: NamedTuple, params: CournotEnvParams) -> dict:
    opt_quantity = CournotGame.nash_policy(params)
    average_quantity = (traj1.actions + traj2.actions) / 2

    return {
        "quantity/1": jnp.mean(traj1.actions),
        "quantity/2": jnp.mean(traj2.actions),
        "quantity/average": jnp.mean(average_quantity),
        # How strongly do the joint actions deviate from the optimal quantity?
        # Since the reward is a linear function of the quantity there is no need to consider it separately.
        "quantity/loss": jnp.mean((opt_quantity / 2 - average_quantity) ** 2),
        "opt_quantity": opt_quantity,
    }
