# Learning with Opponent-Learning Awareness (LOLA) implementation in JAX
# https://arxiv.org/pdf/1709.04326.pdf

from typing import NamedTuple, Any

from dm_env import TimeStep
import haiku as hk
import jax
import jax.numpy as jnp


class TrainingState(NamedTuple):
    params: hk.Params
    random_key: jnp.ndarray


class LOLA:
    """Implements LOLA with exact value functions"""

    def __init__(self, network: hk.Params, random_key: jnp.ndarray):
        def policy(
            params: hk.Params, observation: jnp.ndarray, state: TrainingState
        ):
            """Determines how to choose an action"""
            key, subkey = jax.random.split(state.random_key)
            logits = network.apply(params, observation)
            print("Logits from policy", logits)
            print("Softmax of logits from policy", jax.nn.softmax(logits))
            actions = jax.random.categorical(subkey, logits)
            state = state._replace(random_key=key)
            return int(actions), state

        def loss():
            """Loss function"""
            pass

        def sgd():
            """Stochastic gradient descent"""
            pass

        def make_initial_state(key: jnp.ndarray) -> TrainingState:
            """Make initial training state for LOLA"""
            key, subkey = jax.random.split(key)
            dummy_obs = jnp.zeros(shape=(1, 5))
            params = network.init(subkey, dummy_obs)
            return TrainingState(params=params, random_key=key)

        self.state = make_initial_state(random_key)
        self._policy = policy

    def select_action(self, t: TimeStep):
        """Select action based on observation"""
        # Unpack
        params = self.state.params
        state = self.state
        action, self.state = self._policy(params, t.observation, state)
        return action

    def update(
        self,
        t: TimeStep,
        actions: jnp.ndarray,
        t_prime: TimeStep,
        other_agents=None,
    ):
        """Update agent"""
        # an sgd step requires the parameters of the other agent.
        # currently, the runner file doesn't have access to the other agent's gradients
        # we could put the parameters of the agent inside the timestep
        pass


def make_lola(seed: int) -> LOLA:
    """ "Instantiate LOLA"""
    random_key = jax.random.PRNGKey(seed)

    def forward(inputs):
        """Forward pass for LOLA exact"""
        values = hk.Linear(1, with_bias=False)
        return values(inputs)

    network = hk.without_apply_rng(hk.transform(forward))

    return LOLA(network=network, random_key=random_key)


if __name__ == "__main__":
    lola = make_lola(seed=0)
    print(f"LOLA state: {lola.state}")
    timestep = TimeStep(
        step_type=0,
        reward=1,
        discount=1,
        observation=jnp.array([[1, 0, 0, 0, 0]]),
    )
    action = lola.select_action(timestep)
    print("Action", action)
    timestep = TimeStep(
        step_type=0, reward=1, discount=1, observation=jnp.zeros(shape=(1, 5))
    )
    action = lola.select_action(timestep)
    print("Action", action)
