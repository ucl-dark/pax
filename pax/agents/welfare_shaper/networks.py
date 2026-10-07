"""Networks for the Welfare Shaper agent.

Same architecture as M-FOS: three GRU streams (meta, actor, critic).
The meta stream generates per-episode shaping weights (th) that modulate
the actor and critic outputs.  The only difference from vanilla MFOS is
that the *training objective* changes (handled in the runner / agent update),
not the network topology.
"""

from typing import Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp

from pax import utils


class ActorCriticWelfare(hk.Module):
    """Three-stream GRU actor-critic with meta shaping weights."""

    def __init__(self, num_values: int, hidden_size: int, categorical: bool = True):
        super().__init__(name="ActorCriticWelfare")
        self.linear_t_0 = hk.Linear(hidden_size)
        self.linear_a_0 = hk.Linear(hidden_size)
        self.linear_v_0 = hk.Linear(hidden_size)

        self._meta = hk.GRU(hidden_size)
        self._actor = hk.GRU(hidden_size)
        self._critic = hk.GRU(hidden_size)

        self._meta_layer = hk.Linear(hidden_size)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            with_bias=False,
        )
        self._categorical = categorical

    def __call__(self, inputs: jnp.ndarray, state: jnp.ndarray):
        input, th = inputs
        hidden_t, hidden_a, hidden_v = jnp.split(state, 3, axis=-1)

        meta_input = self.linear_t_0(input)
        action_input = self.linear_a_0(input)
        value_input = self.linear_v_0(input)

        # Actor
        action_output, hidden_a = self._actor(action_input, hidden_a)
        logits = self._logit_layer(th * action_output)

        # Critic
        value_output, hidden_v = self._critic(value_input, hidden_v)
        value = jnp.squeeze(self._value_layer(th * value_output), axis=-1)

        # Meta shaping weights
        mfos_output, hidden_t = self._meta(meta_input, hidden_t)
        _current_th = jax.nn.sigmoid(self._meta_layer(mfos_output))

        hidden = jnp.concatenate([hidden_t, hidden_a, hidden_v], axis=-1)
        state = (_current_th, hidden)
        if self._categorical:
            return distrax.Categorical(logits=logits), value, state
        else:
            return distrax.MultivariateNormalDiag(loc=logits), value, state


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_welfare_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(inputs, state):
        net = ActorCriticWelfare(num_actions, hidden_size)
        logits, values, state = net(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_welfare_avg_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(inputs, state):
        net = ActorCriticWelfare(num_actions, hidden_size)
        hidden_t, hidden_a, hidden_v = jnp.split(state, 3, axis=-1)
        avg_hidden_t = jnp.mean(hidden_t, axis=0, keepdims=True).repeat(state.shape[0], axis=0)
        avg_hidden_a = jnp.mean(hidden_a, axis=0, keepdims=True).repeat(state.shape[0], axis=0)
        avg_hidden_v = jnp.mean(hidden_v, axis=0, keepdims=True).repeat(state.shape[0], axis=0)
        hidden_t = 0.5 * hidden_t + 0.5 * avg_hidden_t
        hidden_a = 0.5 * hidden_a + 0.5 * avg_hidden_a
        hidden_v = 0.5 * hidden_v + 0.5 * avg_hidden_v
        state = jnp.concatenate([hidden_t, hidden_a, hidden_v], axis=-1)
        logits, values, state = net(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state


def make_welfare_continuous_network(num_actions: int, hidden_size: int):
    hidden_state = jnp.zeros((1, 3 * hidden_size))

    def forward_fn(inputs, state):
        net = ActorCriticWelfare(num_actions, hidden_size, categorical=False)
        logits, values, state = net(inputs, state)
        return (logits, values), state

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network, hidden_state
