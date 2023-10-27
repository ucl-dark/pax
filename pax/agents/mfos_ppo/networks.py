from typing import Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from distrax import Categorical

from pax import utils


class ActorCriticMFOS(hk.Module):
    def __init__(self, num_values, hidden_size, categorical=True):
        super().__init__(name="ActorCriticMFOS")
        self.linear_t_0 = hk.Linear(hidden_size)
        self.linear_a_0 = hk.Linear(hidden_size)
        self.linear_v_0 = hk.Linear(hidden_size)

        self._meta = hk.GRU(hidden_size)
        self._actor = hk.GRU(hidden_size)
        self._critic = hk.GRU(hidden_size)

        self._meta_layer = hk.Linear(hidden_size)
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
        self._categorical = categorical

    def __call__(self, inputs: jnp.ndarray, state: jnp.ndarray):
        input, th = inputs
        hidden_t, hidden_a, hidden_v = jnp.split(state, 3, axis=-1)

        # x is simple input
        meta_input = self.linear_t_0(input)
        action_input = self.linear_a_0(input)
        value_input = self.linear_v_0(input)

        # Actor
        action_output, hidden_a = self._actor(action_input, hidden_a)
        logits = self._logit_layer(th * action_output)

        # Critic
        value_output, hidden_v = self._critic(value_input, hidden_v)
        value = jnp.squeeze(self._value_layer(th * value_output), axis=-1)

        # MFOS
        mfos_output, hidden_t = self._meta(meta_input, hidden_t)
        _current_th = jax.nn.sigmoid(self._meta_layer(mfos_output))

        hidden = jnp.concatenate([hidden_t, hidden_a, hidden_v], axis=-1)
        state = (_current_th, hidden)
        if self._categorical:
            return distrax.Categorical(logits=logits), value, state
        else:
            return distrax.MultivariateNormalDiag(loc=logits), value, state


class CNNFusion(hk.Module):
    def __init__(self, output_channels, kernel_shape):
        super().__init__(name="CNN")
        output_channels = output_channels
        kernel_shape = kernel_shape
        self.conv_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear = hk.Linear(output_channels)
        self.flatten = hk.Flatten()

    def __call__(self, inputs):
        obs = inputs["observation"]
        inventory = inputs["inventory"]
        # Actor
        x = self.conv_0(obs)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = jnp.concatenate([x, inventory], axis=-1)
        x = self.linear(x)
        return x


class CNNMFOS(hk.Module):
    def __init__(self, num_values, hidden_size, output_channels, kernel_shape):
        super().__init__(name="CNNMFOS")
        self.t_network = CNNFusion(output_channels, kernel_shape)
        self.a_network = CNNFusion(output_channels, kernel_shape)
        self.v_network = CNNFusion(output_channels, kernel_shape)

        self._meta = hk.GRU(hidden_size)
        self._actor = hk.GRU(hidden_size)
        self._critic = hk.GRU(hidden_size)

        self._meta_layer = hk.Linear(hidden_size)
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

    def __call__(self, inputs: Tuple[dict, jnp.array], state: jnp.ndarray):
        input, th = inputs
        hidden_t, hidden_a, hidden_v = jnp.split(state, 3, axis=-1)

        # x is simple input
        meta_input = self.t_network(input)
        action_input = self.a_network(input)
        value_input = self.v_network(input)

        # Actor
        action_output, hidden_a = self._actor(action_input, hidden_a)
        logits = self._logit_layer(th * action_output)

        # Critic
        value_output, hidden_v = self._critic(value_input, hidden_v)
        value = jnp.squeeze(self._value_layer(th * value_output), axis=-1)

        # MFOS
        mfos_output, hidden_t = self._meta(meta_input, hidden_t)
        _current_th = jax.nn.sigmoid(self._meta_layer(mfos_output))

        hidden = jnp.concatenate([hidden_t, hidden_a, hidden_v], axis=-1)
        state = (_current_th, hidden)
        return (distrax.Categorical(logits=logits), value, state)


def make_mfos_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(
        inputs: jnp.ndarray,
        state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        mfos = ActorCriticMFOS(num_actions, hidden_size)
        logits, values, state = mfos(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state

def make_mfos_avg_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(
        inputs: jnp.ndarray,
        state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        mfos = ActorCriticMFOS(num_actions, hidden_size)
        hidden_t, hidden_a, hidden_v = jnp.split(state, 3, axis=-1)
        avg_hidden_t = jnp.mean(hidden_t, axis=0, keepdims=True).repeat(state.shape[0], axis=0)
        avg_hidden_a = jnp.mean(hidden_a, axis=0, keepdims=True).repeat(state.shape[0], axis=0)
        avg_hidden_v = jnp.mean(hidden_v, axis=0, keepdims=True).repeat(state.shape[0], axis=0)
        hidden_t = 0.5*hidden_t + 0.5*avg_hidden_t
        hidden_a = 0.5*hidden_a + 0.5*avg_hidden_a
        hidden_v = 0.5*hidden_v + 0.5*avg_hidden_v
        state = jnp.concatenate([hidden_t, hidden_a, hidden_v], axis=-1)
        logits, values, state = mfos(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_mfos_continuous_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(
        inputs: jnp.ndarray,
        state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        mfos = ActorCriticMFOS(num_actions, hidden_size, categorical=False)
        logits, values, state = mfos(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_mfos_ipditm_network(
    num_actions: int, hidden_size: int, output_channels, kernel_shape
):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(
        inputs: jnp.ndarray,
        state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> Tuple[Tuple[Categorical, jnp.ndarray], jnp.ndarray]:
        mfos = CNNMFOS(num_actions, hidden_size, output_channels, kernel_shape)
        logits, values, state = mfos(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def test_GRU():
    key = jax.random.PRNGKey(seed=0)
    num_actions = 2
    obs_spec = (5,)
    hidden_size = 3
    key, subkey = jax.random.split(key)
    dummy_obs = jnp.zeros(shape=obs_spec)
    dummy_obs = utils.add_batch_dim(dummy_obs)
    dummy_meta = jnp.zeros(shape=hidden_size)
    dummy_meta = utils.add_batch_dim(dummy_meta)
    dummy_input = (dummy_obs, dummy_meta)

    network, hidden = make_mfos_network(num_actions, hidden_size)

    initial_params = network.init(subkey, dummy_input, hidden)
    print(initial_params.keys())
    print("GRU w_i", initial_params["ActorCriticMFOS/~/gru_1"]["w_i"].shape)
    print("GRU w_h", initial_params["ActorCriticMFOS/~/gru_1"]["w_h"].shape)
    print(
        "Policy head",
        initial_params["ActorCriticMFOS/~/linear_5"]["w"].shape,
    )
    print(
        "Value head",
        initial_params["ActorCriticMFOS/~/linear_3"]["w"].shape,
    )
    inputs = (jnp.zeros(shape=(1, 5)), jnp.zeros(shape=(1, hidden_size)))
    inputs = (jnp.zeros(shape=(10, 5)), jnp.zeros(shape=(10, hidden_size)))
    (logits, values), hidden = network.apply(initial_params, inputs, hidden)
    print(hidden[0].shape)
    return network


if __name__ == "__main__":
    test_GRU()
