from time import time as tic
from typing import Mapping, NamedTuple
import optax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


# class TrainingState(NamedTuple):
#     """Training state consists of network parameters, optimiser state, random key, timesteps, and extras."""

#     params: hk.Params
#     opt_state: optax.GradientTransformation
#     random_key: jnp.ndarray
#     timesteps: int
#     extras: Mapping[str, jnp.ndarray]
#     hidden: None


class TrainingState(NamedTuple):
    """Training state consists of network parameters, optimiser state, random key, timesteps, and extras."""

    policy_params: hk.Params
    value_params: hk.Params
    policy_opt_state: optax.GradientTransformation
    value_opt_state: optax.GradientTransformation
    random_key: jnp.ndarray
    timesteps: int
    extras: Mapping[str, jnp.ndarray]
    hidden: None


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


def copy_state_and_network(agent):
    import copy

    """Copies an agent state and returns the state"""
    state = TrainingState(
        params=copy.deepcopy(agent._state.params),
        opt_state=agent._state.opt_state,
        random_key=agent._state.random_key,
        timesteps=agent._state.timesteps,
        extras={
            "values": jnp.zeros(agent._num_envs),
            "log_probs": jnp.zeros(agent._num_envs),
        },
        hidden=None,
    )
    network = agent.network
    return state, network


def copy_extended_state_and_network(agent):
    import copy

    """Copies an agent state and returns the state"""
    state = TrainingState(
        policy_params=copy.deepcopy(agent._state.policy_params),
        value_params=copy.deepcopy(agent._state.value_params),
        policy_opt_state=agent._state.policy_opt_state,
        value_opt_state=agent._state.value_opt_state,
        random_key=agent._state.random_key,
        timesteps=agent._state.timesteps,
        extras={
            "values": jnp.zeros(agent._num_envs),
            "log_probs": jnp.zeros(agent._num_envs),
        },
        hidden=None,
    )
    policy_network = agent.policy_network
    value_network = agent.value_network
    return state, policy_network, value_network
