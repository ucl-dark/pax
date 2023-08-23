from functools import partial
from typing import NamedTuple

import jax
from jax import numpy as jnp


@partial(jax.jit, static_argnums=(1,))
def rice_stats(traj: NamedTuple, num_players: int) -> dict:
    # obs shape: num_outer_steps x num_inner_steps x num_opponents x num_envs x obs_dim
    result = {
        "temperature": jnp.mean(traj.observations[..., 2]),
        "land_temperature": jnp.mean(traj.observations[..., 3]),
        "carbon_atmosphere": jnp.mean(traj.observations[..., 4]),
        "carbon_upper_ocean": jnp.mean(traj.observations[..., 5]),
        "carbon_land": jnp.mean(traj.observations[..., 6]),
        # Omitted: exogenous_emissions, land_emissions
    }

    region_vars = [
        "labor",
        "capital",
        "gross_output",
        "consumption",
        "investment",
        "balance",
    ]

    offset = 9
    for i, label in enumerate(region_vars):
        start = offset + i * num_players
        end = offset + (i + 1) * num_players
        result[label] = jnp.mean(traj.observations[..., start:end])

    result["final_temperature"] = jnp.sum(traj.dones * traj.observations[..., 2]) / jnp.sum(traj.dones)

    return result


@partial(jax.jit, static_argnums=(1,))
def rice_sarl_stats(traj: NamedTuple, num_players: int) -> dict:
    # obs shape: num_steps x num_envs x obs_dim
    result = {
        # Actions
        "savings_rate": jnp.mean(traj.actions[..., 0]),
        "mitigation_rate": jnp.mean(traj.actions[..., 1]),

        "temperature": jnp.mean(traj.observations[..., 1]),
        "land_temperature": jnp.mean(traj.observations[..., 2]),
        "carbon_atmosphere": jnp.mean(traj.observations[..., 3]),
        "carbon_upper_ocean": jnp.mean(traj.observations[..., 4]),
        "carbon_land": jnp.mean(traj.observations[..., 5]),
        "exogenous_emissions": jnp.mean(traj.observations[..., 6]),
        "land_emissions": jnp.mean(traj.observations[..., 7]),
    }

    region_vars = [
        "labor",
        "capital",
        "gross_output",
        "consumption",
        "investment",
        "balance",
        "production_factor",
        "intensity",
        "mitigation_cost",
        "damages",
        "abatement_cost",
        "production",
        "utility",
        "social_welfare",
    ]

    offset = 8
    for i, label in enumerate(region_vars):
        start = offset + i * num_players
        end = offset + (i + 1) * num_players
        result[label] = jnp.mean(traj.observations[..., start:end])

    result["final_temperature"] = jnp.sum(traj.dones * traj.observations[..., 1]) / jnp.sum(traj.dones)

    return result
