from typing import Optional, Any, Callable, Tuple, NamedTuple

from pax import utils

import distrax
import haiku as hk
import jax
import jax.numpy as jnp


class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            # w_init=hk.initializers.Orthogonal(0.01),  # baseline
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            # w_init=hk.initializers.Orthogonal(1.0),  # baseline
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Categorical(logits=logits), value)


class ContinuousValueHead(hk.Module):
    """Network head that produces a continuous distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),  # baseline
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),  # baseline
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.MultivariateNormalDiag(loc=logits), value)


def make_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                CategoricalValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_cartpole_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

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
                CategoricalValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_GRU(num_actions: int):
    hidden_size = 25
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        gru = hk.GRU(hidden_size)
        embedding, state = gru(inputs, state)
        logits, values = CategoricalValueHead(num_actions)(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return network, hidden_state


def make_GRU_cartpole_network(num_actions: int):
    hidden_size = 256
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """forward function"""
        torso = hk.nets.MLP(
            [hidden_size, hidden_size],
            w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
            b_init=hk.initializers.Constant(0),
            activate_final=True,
        )
        gru = hk.GRU(hidden_size)
        embedding = torso(inputs)
        embedding, state = gru(embedding, state)
        logits, values = CategoricalValueHead(num_actions)(embedding)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))

    return network, hidden_state


def test_GRU():
    key = jax.random.PRNGKey(seed=0)
    num_actions = 2
    obs_spec = (5,)
    key, subkey = jax.random.split(key)
    dummy_obs = jnp.zeros(shape=obs_spec)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    network, hidden = make_GRU(num_actions)
    print(hidden.shape)
    initial_params = network.init(subkey, dummy_obs, hidden)
    print("GRU w_i", initial_params["gru"]["w_i"].shape)
    print("GRU w_h", initial_params["gru"]["w_h"].shape)
    print(
        "Policy head",
        initial_params["categorical_value_head/~/linear"]["w"].shape,
    )
    print(
        "Value head",
        initial_params["categorical_value_head/~/linear_1"]["w"].shape,
    )
    observation = jnp.zeros(shape=(1, 5))
    observation = jnp.zeros(shape=(10, 5))
    (logits, values), hidden = network.apply(
        initial_params, observation, hidden
    )
    print(hidden.shape)
    return network


if __name__ == "__main__":
    test_GRU()
