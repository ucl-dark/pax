from typing import Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp

from pax import utils


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
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Categorical(logits=logits), value)


class CategoricalValueHead_separate(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0.5),
            with_bias=False,
        )

    def __call__(self, inputs: Tuple[jnp.ndarray, jnp.ndarray]):
        action_output, value_output = inputs
        logits = self._logit_layer(action_output)
        value = jnp.squeeze(self._value_layer(value_output), axis=-1)
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


class CNN(hk.Module):
    def __init__(self, args):
        super().__init__(name="CNN")
        output_channels = args.ppo.output_channels
        kernel_shape = args.ppo.kernel_shape
        self.conv_a_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.conv_a_1 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear_a_0 = hk.Linear(output_channels)

        self.flatten = hk.Flatten()

    def __call__(self, inputs: jnp.ndarray):
        # Actor and Critic
        x = self.conv_a_0(inputs)
        x = jax.nn.relu(x)
        x = self.conv_a_1(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear_a_0(x)
        x = jax.nn.relu(x)

        return x


class CNN_separate(hk.Module):
    def __init__(self, args):
        super().__init__(name="CNN")
        output_channels = args.ppo.output_channels
        kernel_shape = args.ppo.kernel_shape
        self.conv_a_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.conv_a_1 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear_a_0 = hk.Linear(output_channels)

        self.conv_v_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.conv_v_1 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear_v_0 = hk.Linear(output_channels)

        self.flatten = hk.Flatten()

    def __call__(self, inputs: jnp.ndarray):
        # Actor
        x = self.conv_a_0(inputs)
        x = jax.nn.relu(x)
        x = self.conv_a_1(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear_a_0(x)
        logits = jax.nn.relu(x)

        # Critic
        x = self.conv_v_0(inputs)
        x = jax.nn.relu(x)
        x = self.conv_v_1(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear_v_0(x)
        val = jax.nn.relu(x)
        return (logits, val)


def make_coingame_network(num_actions: int, network_args):
    def forward_fn(inputs):
        layers = []
        if network_args.ppo.separate:
            cnn = CNN_separate(network_args)
            cvh = CategoricalValueHead_separate(num_values=num_actions)
        else:
            cnn = CNN(network_args)
            cvh = CategoricalValueHead(num_values=num_actions)
        layers.extend([cnn, cvh])
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_GRU_coingame_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, hidden_size))

    def forward_fn(
        inputs: jnp.ndarray, state: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        x = CNN()(inputs)
        gru = hk.GRU(hidden_size)
        embedding, state = gru(x, state)
        logits, values = CategoricalValueHead(num_actions)(embedding)

        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_ipd_network(num_actions: int):
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


def make_GRU_ipd_network(num_actions: int):
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
    network, hidden = make_GRU_ipd_network(num_actions)
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
