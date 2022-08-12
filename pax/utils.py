from time import time as tic
from typing import Mapping, NamedTuple
from functools import partial

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle


class Section(object):
    """
    Examples
    --------
    >>> with Section('Loading Data'):
    >>>     num_samples = load_data()
    >>>     print(f'=> {num_samples} samples loaded')
    will print out something like
    => Loading data ...
    => 42 samples loaded
    => Done!
    """

    def __init__(self, description, newline=True, logger=None, timing="auto"):
        super(Section, self).__init__()
        self.description = description
        self.newline = newline
        self.logger = logger
        self.timing = timing
        self.t0 = None
        self.t1 = None

    def __enter__(self):
        self.t0 = tic()
        self.print_message("=> " + str(self.description) + " ...")

    def __exit__(self, type, value, traceback):
        self.t1 = tic()
        msg = "=> Done"
        if self.timing != "none":
            t, unit = self._get_time_and_unit(self.t1 - self.t0)
            msg += f" in {t:.3f} {unit}"
        self.print_message(msg)
        if self.newline:
            self.print_message()

    def print_message(self, message=""):
        #  if self.logger is None:
        #  try:
        #  print(message, flush=True)
        #  except TypeError:
        #  print(message)
        #  sys.stdout.flush()
        #  else:
        self.logger.info(message)

    def _get_time_and_unit(self, time):
        if self.timing == "auto":
            return self._auto_determine_time_and_unit(time)
        elif self.timing == "us":
            return time * 1000000, "us"
        elif self.timing == "ms":
            return time * 1000, "ms"
        elif self.timing == "s":
            return time, "s"
        elif self.timing == "m":
            return time / 60, "m"
        elif self.timing == "h":
            return time / 3600, "h"
        else:
            raise ValueError(f"Unknown timing mode: {self.timing}")

    def _auto_determine_time_and_unit(self, time):
        if time < 0.001:
            return time * 1000000, "us"
        elif time < 1.0:
            return time * 1000, "ms"
        elif time < 60:
            return time, "s"
        elif time < 3600:
            return time / 60, "m"
        else:
            return time / 3600, "h"


def add_batch_dim(values):
    return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), values)


def to_numpy(values):
    return jax.tree_map(np.asarray, values)


class TrainingState(NamedTuple):
    """Training state consists of network parameters, optimiser state, random key, timesteps"""

    params: hk.Params
    opt_state: optax.GradientTransformation
    random_key: jnp.ndarray
    timesteps: int


class MemoryState(NamedTuple):
    """State consists of network extras (to be batched)"""

    hidden: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


def get_advantages(carry, transition):
    gae, next_value, gae_lambda = carry
    value, reward, discounts = transition
    value_diff = discounts * next_value - value
    delta = reward + value_diff
    gae = delta + discounts * gae_lambda * gae
    return (gae, value, gae_lambda), gae


class ESLog(object):
    def __init__(
        self, num_dims: int, num_generations: int, top_k: int, maximize: bool
    ):
        """Simple jittable logging tool for ES rollouts."""
        self.num_dims = num_dims
        self.num_generations = num_generations
        self.top_k = top_k
        self.maximize = maximize

    @partial(jax.jit, static_argnums=(0,))
    def initialize(self) -> chex.ArrayTree:
        """Initialize the logger storage."""
        log = {
            "top_fitness": jnp.zeros(self.top_k)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_params": jnp.zeros((self.top_k, self.num_dims))
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_1": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_gen_fitness": jnp.zeros(self.top_k)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "top_gen_params": jnp.zeros((self.top_k, self.num_dims))
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_1": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_gen_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_top_gen_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_mean": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "log_gen_std": jnp.zeros(self.num_generations)
            - 1e10 * self.maximize
            + 1e10 * (1 - self.maximize),
            "gen_counter": 0,
        }
        return log

    # @partial(jax.jit, static_argnums=(0,))
    def update(
        self, log: chex.ArrayTree, x: chex.Array, fitness: chex.Array
    ) -> chex.ArrayTree:
        """Update the logging storage with newest data."""
        # Check if there are solutions better than current archive
        def get_top_idx(maximize: bool, vals: jnp.ndarray) -> jnp.ndarray:
            top_idx = maximize * ((-1) * vals).argsort() + (
                (1 - maximize) * vals.argsort()
            )
            return top_idx

        vals = jnp.hstack([log["top_fitness"], fitness])
        params = jnp.vstack([log["top_params"], x])
        top_idx = get_top_idx(self.maximize, vals)
        log["top_fitness"] = vals[top_idx[: self.top_k]]
        log["top_params"] = params[top_idx[: self.top_k]]
        log["log_top_1"] = (
            log["log_top_1"].at[log["gen_counter"]].set(log["top_fitness"][0])
        )
        log["log_top_mean"] = (
            log["log_top_mean"]
            .at[log["gen_counter"]]
            .set(jnp.mean(log["top_fitness"]))
        )
        log["log_top_std"] = (
            log["log_top_std"]
            .at[log["gen_counter"]]
            .set(jnp.std(log["top_fitness"]))
        )
        top_idx = get_top_idx(self.maximize, fitness)
        log["top_gen_fitness"] = fitness[top_idx[: self.top_k]]
        log["top_gen_params"] = x[top_idx[: self.top_k]]
        log["log_gen_1"] = (
            log["log_gen_1"]
            .at[log["gen_counter"]]
            .set(
                self.maximize * jnp.max(fitness)
                + (1 - self.maximize) * jnp.min(fitness)
            )
        )
        log["log_top_gen_mean"] = (
            log["log_top_gen_mean"]
            .at[log["gen_counter"]]
            .set(jnp.mean(log["top_gen_fitness"]))
        )
        log["log_top_gen_std"] = (
            log["log_top_gen_std"]
            .at[log["gen_counter"]]
            .set(jnp.std(log["top_gen_fitness"]))
        )
        log["log_gen_mean"] = (
            log["log_gen_mean"].at[log["gen_counter"]].set(jnp.mean(fitness))
        )
        log["log_gen_std"] = (
            log["log_gen_std"].at[log["gen_counter"]].set(jnp.std(fitness))
        )
        log["gen_counter"] += 1
        return log

    def save(self, log: chex.ArrayTree, filename: str):
        """Save different parts of logger in .pkl file."""
        with open(filename, "wb") as handle:
            pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str):
        """Reload the pickle logger and return dictionary."""
        with open(filename, "rb") as handle:
            es_logger = pickle.load(handle)
        return es_logger

    def plot(
        self,
        log,
        title,
        ylims=None,
        fig=None,
        ax=None,
        no_legend=False,
    ):
        """Plot fitness trajectory from evo logger over generations."""
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        int_range = jnp.arange(1, log["gen_counter"] + 1)
        ax.plot(
            int_range, log["log_top_1"][: log["gen_counter"]], label="Top 1"
        )
        ax.plot(
            int_range,
            log["log_top_mean"][: log["gen_counter"]],
            label=f"Top-{self.top_k} Mean",
        )
        ax.plot(
            int_range, log["log_gen_1"][: log["gen_counter"]], label="Gen. 1"
        )
        ax.plot(
            int_range,
            log["log_gen_mean"][: log["gen_counter"]],
            label="Gen. Mean",
        )
        if ylims is not None:
            ax.set_ylim(ylims)
        if not no_legend:
            ax.legend()
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Number of Generations")
        ax.set_ylabel("Fitness Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return fig, ax
