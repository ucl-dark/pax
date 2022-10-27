import jax.numpy as jnp
import jax
import pytest

from pax.envs.infinite_matrix_game import InfiniteMatrixGame, EnvParams

from pax.strategies import TitForTat


def test_single_infinite_game():
    rng = jax.random.PRNGKey(0)
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    alt_policy = 20 * jnp.ones((5))
    def_policy = -20 * jnp.ones((5))
    tft_policy = 20 * jnp.array([1.0, -1.0, 1.0, -1.0, 1.0])

    # discount of 0.99 -> 1/(0.001) ~ 100 timestep
    env = InfiniteMatrixGame()
    env_params = EnvParams(payoff_matrix=payoff, num_steps=10, gamma=0.99)

    obs, env_state = env.reset(rng, env_params)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (alt_policy, alt_policy), env_params
    )

    assert rewards[0] == rewards[1]
    assert jnp.isclose(2, rewards[0], atol=0.01)
    assert jnp.allclose(obs[0], jnp.ones((10)), atol=0.01)
    assert jnp.allclose(obs[1], jnp.ones((10)), atol=0.01)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (def_policy, def_policy), env_params
    )

    assert rewards[0] == rewards[1]
    assert jnp.isclose(1, rewards[0], atol=0.01)
    assert jnp.allclose(obs[0], jnp.zeros((10)))
    assert jnp.allclose(obs[1], jnp.zeros((10)))

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (def_policy, alt_policy), env_params
    )
    assert rewards[0] != rewards[1]
    assert jnp.isclose(3, rewards[0])
    assert jnp.isclose(0.0, rewards[1], atol=0.0001)
    assert jnp.allclose(obs[0], jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    assert jnp.allclose(obs[1], jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]))

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (alt_policy, tft_policy), env_params
    )
    assert jnp.isclose(2, rewards[0])
    assert jnp.isclose(2, rewards[1])
    assert jnp.allclose(obs[0], jnp.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1]))
    assert jnp.allclose(obs[1], jnp.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1]))

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (tft_policy, tft_policy), env_params
    )
    assert jnp.isclose(2, rewards[0])
    assert jnp.isclose(2, rewards[1])
    assert jnp.allclose(obs[0], jnp.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]))
    assert jnp.allclose(obs[1], jnp.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1]))

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (tft_policy, def_policy), env_params
    )
    assert jnp.isclose(0.99, rewards[0])
    assert jnp.isclose(1.02, rewards[1])
    assert jnp.allclose(obs[0], jnp.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))
    assert jnp.allclose(obs[1], jnp.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1]))

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (def_policy, tft_policy), env_params
    )
    assert jnp.isclose(1.02, rewards[0])
    assert jnp.isclose(0.99, rewards[1])
    assert jnp.allclose(obs[0], jnp.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1]))
    assert jnp.allclose(obs[1], jnp.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0]))


def test_batch_infinite_game():
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    rng = jax.random.PRNGKey(0)
    alt_policy = 20 * jnp.ones((1, 5))
    def_policy = -20 * jnp.ones((1, 5))
    tft_policy = 20 * jnp.array([[1, -1, 1, -1, 1]])

    env = InfiniteMatrixGame()
    env_params = EnvParams(payoff_matrix=payoff, num_steps=10, gamma=0.99)

    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    rngs = jnp.concatenate([rng, rng, rng], axis=0).reshape(3, -1)

    batched_alt = jnp.concatenate([alt_policy, alt_policy, alt_policy], axis=0)
    batched_def = jnp.concatenate([def_policy, def_policy, def_policy], axis=0)
    batch_mixed_1 = jnp.concatenate(
        [alt_policy, def_policy, tft_policy], axis=0
    )
    batch_mixed_2 = jnp.concatenate(
        [def_policy, tft_policy, tft_policy], axis=0
    )

    # first step
    obs, env_state = env.reset(rngs, env_params)

    obs, env_state, rewards, done, info = env.step(
        rngs, env_state, (batched_alt, batched_alt), env_params
    )
    assert jnp.allclose(rewards[0], rewards[1])
    assert jnp.allclose(jnp.array([2, 2, 2]), rewards[0], atol=0.01)

    obs, env_state, rewards, done, info = env.step(
        rngs, env_state, (batched_alt, batched_def), env_params
    )
    assert jnp.allclose(jnp.array([0, 0, 0]), rewards[0], atol=0.01)
    assert jnp.allclose(jnp.array([3, 3, 3]), rewards[1], atol=0.01)

    obs, env_state, rewards, done, info = env.step(
        rngs, env_state, (batch_mixed_1, batch_mixed_2), env_params
    )
    assert jnp.allclose(jnp.array([0.0, 1.02, 2]), rewards[0])
    assert jnp.allclose(jnp.array([3, 0.99, 2]), rewards[1])

    obs, env_state, rewards, done, info = env.step(
        rngs, env_state, (batch_mixed_2, batch_mixed_1), env_params
    )
    assert jnp.allclose(jnp.array([3, 0.99, 2]), rewards[0], atol=0.01)
    assert jnp.allclose(jnp.array([0, 1.02, 2]), rewards[1], atol=0.01)
