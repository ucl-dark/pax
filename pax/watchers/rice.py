from typing import NamedTuple

import jax
import numpy as np
import wandb
from jax import numpy as jnp


def rice_stats(traj: NamedTuple, num_players: int) -> dict:
    # obs shape: num_outer_steps x num_inner_steps x num_opponents x num_envs x obs_dim
    global_temperature = traj.observations[..., 2]
    carbon_atmosphere = traj.observations[..., 3]
    carbon_upper_ocean = traj.observations[..., 4]
    carbon_land = traj.observations[..., 5]
    actions = observations[..., :num_players]


    for i in range(num_players):
        stats["fishery/effort_" + str(i)] = jnp.mean(observations[..., i])

    return stats

