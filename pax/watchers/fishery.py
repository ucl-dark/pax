from typing import NamedTuple

import jax
import numpy as np
import wandb
from jax import numpy as jnp


def fishery_stats(observations: jnp.ndarray, num_players: int) -> dict:
    # obs shape: num_outer_steps x num_inner_steps x num_opponents x num_envs x obs_dim
    stock_obs = observations[..., -1]
    actions = observations[..., :num_players]
    # TODO this blows up the memory usage
    # flattened_stock_obs = jnp.ravel(stock_obs)
    # split_stock_obs = jnp.array(jnp.split(flattened_stock_obs, flattened_stock_obs.shape[0] // num_inner_steps))
    stats = {
        "fishery/stock": jnp.mean(stock_obs),
        "fishery/mean_effort": actions.mean()
    }

    for i in range(num_players):
        stats["fishery/effort_" + str(i)] = jnp.mean(observations[..., i])

    return stats


def fishery_eval_stats(traj1: NamedTuple, traj2: NamedTuple) -> dict:
    # Calculate effort for both agents
    effort_1 = jax.nn.sigmoid(traj1.actions).squeeze().tolist()
    effort_2 = jax.nn.sigmoid(traj2.actions).squeeze().tolist()
    ep_length = np.arange(len(effort_1)).tolist()

    # Plot the effort on the same graph
    effort_plot = wandb.plot.line_series(xs=ep_length, ys=[effort_1, effort_2],
                                         keys=["effort_1", "effort_2"],
                                         xname="step",
                                         title="Agent effort over one episode")

    stock_obs = traj1.observations[..., 0].squeeze().tolist()
    stock_table = wandb.Table(data=[[x, y] for (x, y) in zip(ep_length, stock_obs)], columns=["step", "stock"])
    # Plot the stock in a separate graph
    stock_plot = wandb.plot.line(stock_table, x="step", y="stock", title="Stock over one episode")

    return {
        "fishery/stock": stock_plot,
        "fishery/effort": effort_plot
    }
