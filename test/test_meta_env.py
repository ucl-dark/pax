import jax.numpy as jnp

from pax.meta_env import InfiniteMatrixGame

# payoff matrices for four games
ipd = [[2, 2], [0, 3], [3, 0], [1, 1]]
stag = [[4, 4], [1, 3], [3, 1], [2, 2]]
sexes = [[3, 2], [0, 0], [0, 0], [2, 3]]
chicken = [[0, 0], [-1, 1], [1, -1], [-2, -2]]
test_payoffs = [ipd, stag, sexes, chicken]


def test_infinite_game():
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    # discount of 0.99 -> 1/(0.001) ~ 100 timestep

    env = InfiniteMatrixGame(1, payoff, 10, 0.99, 0)
    alt_policy = 20 * jnp.ones((1, 5))
    def_policy = -20 * jnp.ones((1, 5))
    tft_policy = 20 * jnp.array([[1, -1, 1, -1, 1]])

    # first step
    t0, t1 = env.step((alt_policy, def_policy))

    # assert jnp.allclose(t0.observation, 0.5 * jnp.ones((1, 10)))
    # assert jnp.allclose(t1.observation, 0.5 * jnp.ones((1, 10)))

    t0, t1 = env.step([alt_policy, alt_policy])
    assert t0.reward == t1.reward
    assert jnp.isclose(2, t0.reward, atol=0.01)
    assert jnp.allclose(t0.observation, jnp.ones((1, 10)), atol=0.01)
    assert jnp.allclose(t1.observation, jnp.ones((1, 10)), atol=0.01)

    t0, t1 = env.step([def_policy, def_policy])
    assert t0.reward == t1.reward
    assert jnp.isclose(1, t0.reward, atol=0.01)
    assert jnp.allclose(t0.observation, jnp.zeros((1, 10)))
    assert jnp.allclose(t1.observation, jnp.zeros((1, 10)))

    t0, t1 = env.step([def_policy, alt_policy])
    assert t0.reward != t1.reward
    assert jnp.isclose(3, t0.reward)
    assert jnp.isclose(0.0, t1.reward, atol=0.0001)
    assert jnp.allclose(
        t0.observation, jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    )
    assert jnp.allclose(
        t1.observation, jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    )

    t0, t1 = env.step([alt_policy, tft_policy])
    assert jnp.isclose(2, t0.reward)
    assert jnp.isclose(2, t1.reward)
    assert jnp.allclose(
        t0.observation, jnp.array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1])
    )
    assert jnp.allclose(
        t1.observation, jnp.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    )

    t0, t1 = env.step([tft_policy, tft_policy])
    assert jnp.isclose(2, t0.reward)
    assert jnp.isclose(2, t1.reward)
    assert jnp.allclose(
        t0.observation, jnp.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    )
    assert jnp.allclose(
        t1.observation, jnp.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    )

    t0, t1 = env.step([tft_policy, def_policy])
    assert jnp.isclose(0.99, t0.reward)
    assert jnp.isclose(1.02, t1.reward)
    assert jnp.allclose(
        t0.observation, jnp.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    )
    assert jnp.allclose(
        t1.observation, jnp.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1])
    )

    t0, t1 = env.step([def_policy, tft_policy])
    assert jnp.isclose(1.02, t0.reward)
    assert jnp.isclose(0.99, t1.reward)
    assert jnp.allclose(
        t0.observation, jnp.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 1])
    )
    assert jnp.allclose(
        t1.observation, jnp.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    )


def test_batch_infinite_game():
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    # discount of 0.99 -> 1/(0.001) ~ 100 timestep

    env = InfiniteMatrixGame(3, payoff, 10, 0.99, 0)
    alt_policy = 20 * jnp.ones((1, 5))
    def_policy = -20 * jnp.ones((1, 5))
    tft_policy = 20 * jnp.array([[1, -1, 1, -1, 1]])

    batched_alt = jnp.concatenate([alt_policy, alt_policy, alt_policy], axis=0)
    batched_def = jnp.concatenate([def_policy, def_policy, def_policy], axis=0)
    batch_mixed_1 = jnp.concatenate(
        [alt_policy, def_policy, tft_policy], axis=0
    )
    batch_mixed_2 = jnp.concatenate(
        [def_policy, tft_policy, tft_policy], axis=0
    )

    # first step
    tstep_0, tstep_1 = env.step((batched_alt, batched_alt))

    t0, t1 = env.step([batched_alt, batched_alt])
    assert jnp.allclose(t0.reward, t1.reward)
    assert jnp.allclose(jnp.array([2, 2, 2]), t0.reward, atol=0.01)

    t0, t1 = env.step([batched_alt, batched_def])
    assert jnp.allclose(jnp.array([0, 0, 0]), t0.reward, atol=0.01)
    assert jnp.allclose(jnp.array([3, 3, 3]), t1.reward, atol=0.01)

    t0, t1 = env.step([batch_mixed_1, batch_mixed_2])
    assert jnp.allclose(jnp.array([0.0, 1.02, 2]), t0.reward)
    assert jnp.allclose(jnp.array([3, 0.99, 2]), t1.reward)

    t0, t1 = env.step([batch_mixed_2, batch_mixed_1])
    assert jnp.allclose(jnp.array([3, 0.99, 2]), t0.reward, atol=0.01)
    assert jnp.allclose(jnp.array([0, 1.02, 2]), t1.reward, atol=0.01)
