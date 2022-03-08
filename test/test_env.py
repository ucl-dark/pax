import jax.numpy as jnp
import pytest

from pax.env import IteratedPrisonersDilemma
from pax.strategies import TitForTat


def test_single_batch_rewards() -> None:
    num_envs = 1
    env = IteratedPrisonersDilemma(5, num_envs)
    action = jnp.ones((num_envs, 1), dtype=jnp.int32)
    r_array = jnp.ones((num_envs, 1), dtype=jnp.int32)

    # first step
    tstep_0, tstep_1 = env.step((0 * action, 0 * action))
    assert tstep_0.reward is None
    assert tstep_1.reward is None

    tstep_0, tstep_1 = env.step((0 * action, 0 * action))
    assert jnp.array_equal(tstep_0.reward, 2 * r_array)
    assert jnp.array_equal(tstep_1.reward, 2 * r_array)

    tstep_0, tstep_1 = env.step((1 * action, 0 * action))
    assert jnp.array_equal(tstep_0.reward, 3 * r_array)
    assert jnp.array_equal(tstep_1.reward, 0 * r_array)

    tstep_0, tstep_1 = env.step((0 * action, 1 * action))
    assert jnp.array_equal(tstep_0.reward, 0 * r_array)
    assert jnp.array_equal(tstep_1.reward, 3 * r_array)

    tstep_0, tstep_1 = env.step((1 * action, 1 * action))
    assert jnp.array_equal(tstep_0.reward, 1 * r_array)
    assert jnp.array_equal(tstep_1.reward, 1 * r_array)


testdata = [
    ((0, 0), (2, 2)),
    ((1, 0), (3, 0)),
    ((0, 1), (0, 3)),
    ((1, 1), (1, 1)),
]


@pytest.mark.parametrize("actions, expected_rewards", testdata)
def test_batch_outcomes(actions, expected_rewards) -> None:
    num_envs = 3
    all_ones = jnp.ones((num_envs, 1))

    env = IteratedPrisonersDilemma(5, num_envs)
    env.reset()

    action_1, action_2 = actions
    expected_r1, expected_r2 = expected_rewards

    tstep_0, tstep_1 = env.step((action_1 * all_ones, action_2 * all_ones))
    assert jnp.array_equal(
        tstep_0.reward, expected_r1 * jnp.ones((num_envs, 1))
    )
    assert jnp.array_equal(
        tstep_1.reward, expected_r2 * jnp.ones((num_envs, 1))
    )
    assert tstep_0.last() == False
    assert tstep_1.last() == False


def test_mixed_batched_outcomes() -> None:
    pass


def test_tit_for_tat_match() -> None:
    num_envs = 5
    env = IteratedPrisonersDilemma(5, num_envs)
    t_0, t_1 = env.reset()

    tit_for_tat = TitForTat()

    action_0 = tit_for_tat.select_action(t_0)
    action_1 = tit_for_tat.select_action(t_1)
    assert jnp.array_equal(action_0, action_1)

    t_0, t_1 = env.step((action_0, action_1))
    assert jnp.array_equal(t_0.reward, t_1.reward)


def test_observation() -> None:
    num_envs = 1
    env = IteratedPrisonersDilemma(5, num_envs)
    initial_state = jnp.ones((num_envs, 1))

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
        env._observation(jnp.array([[0], [1], [2], [3], [4]]))[0],
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
        env._observation(jnp.array([[0], [1], [2], [3], [4]]))[1],
    )


def test_done():
    num_envs = 1
    env = IteratedPrisonersDilemma(5, num_envs)
    action = jnp.ones((num_envs, 1))

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


def test_reset():
    num_envs = 1
    env = IteratedPrisonersDilemma(5, num_envs)
    state = jnp.ones((num_envs, 1))

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
