from functools import partial
from typing import NamedTuple, List

import jax
from jax import numpy as jnp

from pax.envs.rice.c_rice import ClubRice
from pax.envs.rice.rice import EnvState, Rice


@partial(jax.jit, static_argnums=(1, 2))
def c_rice_stats(
    trajectories: List[NamedTuple], num_players: int, mediator: bool
) -> dict:
    traj = trajectories[0]
    # obs shape: num_outer_steps x num_inner_steps x num_opponents x num_envs x obs_dim
    result = {
        "club_mitigation_rate": jnp.mean(traj.observations[..., 2]),
        "club_tariff_rate": jnp.mean(traj.observations[..., 3]),
        "temperature": jnp.mean(traj.observations[..., 4]),
        "land_temperature": jnp.mean(traj.observations[..., 5]),
        "carbon_atmosphere": jnp.mean(traj.observations[..., 6]),
        "carbon_upper_ocean": jnp.mean(traj.observations[..., 7]),
        "carbon_land": jnp.mean(traj.observations[..., 8]),
    }

    region_vars = [
        "gross_output",
        "investment",
        "abatement_cost",
        "tariff_revenue",
        "club_membership",
    ]

    offset = 9
    for i, label in enumerate(region_vars):
        start = offset + i * num_players
        end = offset + (i + 1) * num_players
        result[label] = jnp.mean(traj.observations[..., start:end])

    num_episodes = jnp.sum(traj.dones)
    result["final_temperature"] = jnp.where(
        num_episodes != 0,
        jnp.sum(traj.dones * traj.observations[..., 4]) / num_episodes,
        0,
    )

    region_trajectories = (
        trajectories if mediator is False else trajectories[1:]
    )
    total_reward = jnp.array(
        [jnp.sum(_traj.rewards) for _traj in region_trajectories]
    ).sum()
    result["train/total_reward_per_episode"] = jnp.where(
        num_episodes != 0, total_reward / num_episodes, 0
    )

    # Reward per region (necessary to compare weight_sharing against distributed methods)
    for i, _traj in enumerate(region_trajectories):
        result[f"train/total_reward_per_episode_region_{i}"] = jnp.where(
            num_episodes != 0, _traj.rewards.sum() / num_episodes, 0
        )

    return result


ep_length = 20


@partial(jax.jit, static_argnums=(2))
def c_rice_eval_stats(
    trajectories: List[NamedTuple], env_state: EnvState, env: ClubRice
) -> dict:
    # In the stacked env_state the inner steps are one long sequence in the first dimension
    # But to compute step statistics we need it to be episodes x steps x  ...
    ep_count = int(env_state.global_temperature.shape[1] / 20)
    env_state = jax.tree_util.tree_map(
        lambda x: x.reshape((x.shape[0] * ep_count, ep_length, *x.shape[2:])),
        env_state,
    )
    # Output dim: episodes x steps x envs x agents x opps x envs x ...
    # Because the initial obs are not included the timesteps are shifted like so:
    # 1, 2, ..., 19, 0
    # Since the initial obs are always the same we can just shift them to the start
    env_state: EnvState = jax.tree_util.tree_map(
        lambda x: jnp.concatenate((x[:, -1:, ...], x[:, :-1, ...]), axis=1),
        env_state,
    )

    def add_atrib(name, value, axis):
        result[f"states/{name}_mean"] = value.mean(axis=axis)
        result[f"states/{name}_std"] = value.std(axis=axis)
        result[f"states/{name}_min"] = value.min(axis=axis)
        result[f"states/{name}_max"] = value.max(axis=axis)

    result = {}

    add_atrib("temperature", env_state.global_temperature, axis=(0, 2, 3))
    add_atrib("carbon_mass", env_state.global_carbon_mass, axis=(0, 2, 3))
    add_atrib("labor", env_state.labor_all, axis=(0, 2, 3))
    add_atrib("capital", env_state.capital_all, axis=(0, 2, 3))
    add_atrib(
        "production_factor", env_state.production_factor_all, axis=(0, 2, 3)
    )
    add_atrib("intensity", env_state.intensity_all, axis=(0, 2, 3))
    add_atrib("balance", env_state.balance_all, axis=(0, 2, 3))
    add_atrib("gross_output", env_state.gross_output_all, axis=(0, 2, 3))
    add_atrib("investment", env_state.investment_all, axis=(0, 2, 3))
    add_atrib("production", env_state.production_all, axis=(0, 2, 3))
    add_atrib("consumption", env_state.consumption_all, axis=(0, 2, 3))
    add_atrib("abatement_cost", env_state.abatement_cost_all, axis=(0, 2, 3))
    add_atrib("tariff_revenue", env_state.tariff_revenue_all, axis=(0, 2, 3))
    add_atrib("carbon_price", env_state.carbon_price_all, axis=(0, 2, 3))
    add_atrib("club_membership", env_state.club_membership_all, axis=(0, 2, 3))
    add_atrib("utility", env_state.utility_all, axis=(0, 2, 3))
    add_atrib("social_welfare", env_state.social_welfare_all, axis=(0, 2, 3))
    add_atrib("mitigation_cost", env_state.mitigation_cost_all, axis=(0, 2, 3))
    add_atrib("damages", env_state.damages_all, axis=(0, 2, 3))

    actions = jnp.stack([traj.actions for traj in trajectories])
    # n_players x rollouts x steps x ... x n_actions
    # reshape -> n_players x (rollouts * n_episodes) x episode_steps x ... x n_actions
    # transpose -> episode_steps x n_players x (rollouts * n_episodes) x ... x n_actions
    actions = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (x.shape[0], x.shape[1] * ep_count, ep_length, *x.shape[3:])
        ).transpose((2, 0, 1, 3, 4, 5)),
        actions,
    )
    add_atrib(
        "savings_rate", actions[..., env.savings_action_index], axis=(2, 3, 4)
    )
    add_atrib(
        "mitigation_rate",
        actions[..., env.mitigation_rate_action_index],
        axis=(2, 3, 4),
    )
    add_atrib(
        "export_limit", actions[..., env.export_action_index], axis=(2, 3, 4)
    )
    add_atrib(
        "club_join_action",
        actions[..., env.join_club_action_index],
        axis=(2, 3, 4),
    )
    add_atrib(
        "imports",
        actions[
            ...,
            env.desired_imports_action_index : env.desired_imports_action_index
            + env.import_actions_n,
        ],
        axis=(2, 3, 4),
    )
    add_atrib(
        "tariffs",
        actions[
            ...,
            env.tariffs_action_index : env.tariffs_action_index
            + env.tariff_actions_n,
        ],
        axis=(2, 3, 4),
    )

    observations = trajectories[0].observations
    observations = jax.tree_util.tree_map(
        lambda x: x.reshape(
            (x.shape[0] * ep_count, ep_length, *x.shape[2:])
        ).transpose((1, 0, 2, 3, 4)),
        observations,
    )
    club_mitigation_rate = observations[..., 2]
    add_atrib("club_mitigation_rate", club_mitigation_rate, axis=(1, 2, 3))
    add_atrib("club_tariff_rate", observations[..., 3], axis=(1, 2, 3))

    # Mitigation and tariff after climate club applied
    real_mitigation = jnp.where(
        env_state.club_membership_all == 1,
        # Ensure all arrays are of shape (episodes, steps, opps, envs, agents)
        actions[..., env.mitigation_rate_action_index].transpose(
            (2, 0, 3, 4, 1)
        )[..., 0:1],
        actions[..., env.mitigation_rate_action_index].transpose(
            (2, 0, 3, 4, 1)
        )[..., 1:],
    )
    add_atrib(
        "real_mitigation_rate",
        real_mitigation,
        axis=(2, 3, 4),
    )
    add_atrib("real_tariff", env_state.future_tariff, axis=(0, 2, 3))

    return result
