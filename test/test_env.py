import jax.numpy as jnp
import pytest

from pax.env_inner import InfiniteMatrixGame, SequentialMatrixGame
from pax.strategies import TitForTat

# payoff matrices for four games
ipd = [[2, 2], [0, 3], [3, 0], [1, 1]]
stag = [[4, 4], [1, 3], [3, 1], [2, 2]]
sexes = [[3, 2], [0, 0], [0, 0], [2, 3]]
chicken = [[0, 0], [-1, 1], [1, -1], [-2, -2]]
test_payoffs = [ipd, stag, sexes, chicken]


@pytest.mark.parametrize("payoff", test_payoffs)
def test_single_batch_rewards(payoff) -> None:
    num_envs = 1
    env = SequentialMatrixGame(num_envs, payoff, 5)
    action = jnp.ones((num_envs,), dtype=jnp.int32)
    r_array = jnp.ones((num_envs,), dtype=jnp.int32)

    # payoffs
    cc_p1, cc_p2 = payoff[0][0], payoff[0][1]
    cd_p1, cd_p2 = payoff[1][0], payoff[1][1]
    dc_p1, dc_p2 = payoff[2][0], payoff[2][1]
    dd_p1, dd_p2 = payoff[3][0], payoff[3][1]

    # first step
    tstep_0, tstep_1 = env.step((0 * action, 0 * action))
    tstep_0, tstep_1 = env.step((0 * action, 0 * action))
    assert jnp.array_equal(tstep_0.reward, cc_p1 * r_array)
    assert jnp.array_equal(tstep_1.reward, cc_p2 * r_array)

    tstep_0, tstep_1 = env.step((1 * action, 0 * action))
    assert jnp.array_equal(tstep_0.reward, dc_p1 * r_array)
    assert jnp.array_equal(tstep_1.reward, dc_p2 * r_array)

    tstep_0, tstep_1 = env.step((0 * action, 1 * action))
    assert jnp.array_equal(tstep_0.reward, cd_p1 * r_array)
    assert jnp.array_equal(tstep_1.reward, cd_p2 * r_array)

    tstep_0, tstep_1 = env.step((1 * action, 1 * action))
    assert jnp.array_equal(tstep_0.reward, dd_p1 * r_array)
    assert jnp.array_equal(tstep_1.reward, dd_p2 * r_array)


testdata = [
    ((0, 0), (2, 2), ipd),
    ((1, 0), (3, 0), ipd),
    ((0, 1), (0, 3), ipd),
    ((1, 1), (1, 1), ipd),
    ((0, 0), (4, 4), stag),
    ((1, 0), (3, 1), stag),
    ((0, 1), (1, 3), stag),
    ((1, 1), (2, 2), stag),
    ((0, 0), (3, 2), sexes),
    ((1, 0), (0, 0), sexes),
    ((0, 1), (0, 0), sexes),
    ((1, 1), (2, 3), sexes),
    ((0, 0), (0, 0), chicken),
    ((1, 0), (1, -1), chicken),
    ((0, 1), (-1, 1), chicken),
    ((1, 1), (-2, -2), chicken),
]


@pytest.mark.parametrize("actions, expected_rewards, payoff", testdata)
def test_batch_outcomes(actions, expected_rewards, payoff) -> None:
    num_envs = 3
    all_ones = jnp.ones((num_envs,))
    env = SequentialMatrixGame(num_envs, payoff, 5)
    env.reset()

    action_1, action_2 = actions
    expected_r1, expected_r2 = expected_rewards

    tstep_0, tstep_1 = env.step((action_1 * all_ones, action_2 * all_ones))
    assert jnp.array_equal(tstep_0.reward, expected_r1 * jnp.ones((num_envs,)))
    assert jnp.array_equal(tstep_1.reward, expected_r2 * jnp.ones((num_envs,)))
    assert tstep_0.last() == False
    assert tstep_1.last() == False


def test_mixed_batched_outcomes() -> None:
    pass


def test_tit_for_tat_match() -> None:
    num_envs = 5
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = SequentialMatrixGame(num_envs, payoff, 5)
    t_0, t_1 = env.reset()

    tit_for_tat = TitForTat()

    action_0 = tit_for_tat.select_action(t_0)
    action_1 = tit_for_tat.select_action(t_1)
    assert jnp.array_equal(action_0, action_1)

    t_0, t_1 = env.step((action_0, action_1))
    assert jnp.array_equal(t_0.reward, t_1.reward)


def test_observation() -> None:
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = SequentialMatrixGame(num_envs, payoff, 5)
    initial_state = jnp.ones((num_envs,))

    # start
    obs_1, _ = env._observation(4 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 0.0, 0.0, 0.0, 1.0]]),
        obs_1,
    )

    obs_1, _ = env._observation(0 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[1.0, 0.0, 0.0, 0.0, 0.0]]),
        obs_1,
    )

    obs_1, _ = env._observation(1 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 1.0, 0.0, 0.0, 0.0]]),
        obs_1,
    )
    obs_1, _ = env._observation(2 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 0.0, 1.0, 0.0, 0.0]]),
        obs_1,
    )

    obs_1, _ = env._observation(3 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 0.0, 0.0, 1.0, 0.0]]),
        obs_1,
    )

    assert jnp.array_equal(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        env._observation(jnp.array([0, 1, 2, 3, 4]))[0],
    )

    # start
    _, obs_2 = env._observation(4 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 0.0, 0.0, 0.0, 1.0]]),
        obs_2,
    )

    _, obs_2 = env._observation(0 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[1.0, 0.0, 0.0, 0.0, 0.0]]),
        obs_2,
    )

    # 1 -> 2, 2 -> 1
    _, obs_2 = env._observation(1 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 0.0, 1.0, 0.0, 0.0]]),
        obs_2,
    )
    _, obs_2 = env._observation(2 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 1.0, 0.0, 0.0, 0.0]]),
        obs_2,
    )
    _, obs_2 = env._observation(3 * initial_state)
    assert jnp.array_equal(
        jnp.array(num_envs * [[0.0, 0.0, 0.0, 1.0, 0.0]]),
        obs_2,
    )

    assert jnp.array_equal(
        jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        ),
        env._observation(jnp.array([0, 1, 2, 3, 4]))[1],
    )


def test_done():
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = SequentialMatrixGame(num_envs, payoff, 5)
    action = jnp.ones((num_envs,))

    # check first
    t_0, t_1 = env.step((0 * action, 0 * action))
    assert t_0.last() == False
    assert t_1.last() == False

    for _ in range(4):
        t_0, t_1 = env.step((0 * action, 0 * action))
        assert t_0.last() == False
        assert t_1.last() == False

    # check final
    t_0, t_1 = env.step((0 * action, 0 * action))
    assert t_0.last() == True
    assert t_1.last() == True
    # # Testing
    # assert 1 == 2


def test_reset():
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = SequentialMatrixGame(num_envs, payoff, 5)
    state = jnp.ones((num_envs,))

    env.reset()

    for _ in range(4):
        t_0, t_1 = env.step((0 * state, 0 * state))
        assert t_0.last() == False
        assert t_1.last() == False

    env.reset()

    for _ in range(4):
        t_0, t_1 = env.step((0 * state, 0 * state))
        assert t_0.last() == False
        assert t_1.last() == False


def test_infinite_game():
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    # discount of 0.99 -> 1/(0.001) ~ 100 timestep

    env = InfiniteMatrixGame(1, payoff, 10, 0.99, 0)
    alt_policy = 20 * jnp.ones((1, 5))
    def_policy = -20 * jnp.ones((1, 5))
    tft_policy = 20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]])

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
