import jax.numpy as jnp
import jax
import pytest

from pax.envs.iterated_matrix_game import IteratedMatrixGame
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
    rng = jax.random.PRNGKey(0)
    env = IteratedMatrixGame()
    env_state = IteratedMatrixGame.State(payoff, 5, 5)

    obs, env_state = env.env_reset(env_state)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # payoffs
    cc_p1, cc_p2 = payoff[0][0], payoff[0][1]
    cd_p1, cd_p2 = payoff[1][0], payoff[1][1]
    dc_p1, dc_p2 = payoff[2][0], payoff[2][1]
    dd_p1, dd_p2 = payoff[3][0], payoff[3][1]

    # first step
    tstep_0, tstep_1 = env.step(rng, (0 * action, 0 * action), env_state)

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
    env = IteratedMatrixGame(num_envs, payoff, 5, 10)
    env.reset()

    action_1, action_2 = actions
    expected_r1, expected_r2 = expected_rewards

    tstep_0, tstep_1 = env.step((action_1 * all_ones, action_2 * all_ones))

    assert jnp.array_equal(tstep_0.reward, expected_r1 * jnp.ones((num_envs,)))
    assert jnp.array_equal(tstep_1.reward, expected_r2 * jnp.ones((num_envs,)))
    # assert tstep_0.last() == False
    # assert tstep_1.last() == False


def test_mixed_batched_outcomes() -> None:
    pass


def test_tit_for_tat_match() -> None:
    num_envs = 5
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = IteratedMatrixGame(num_envs, payoff, 5, 10)
    t_0, t_1 = env.reset()

    tit_for_tat = TitForTat(num_envs)

    action_0 = tit_for_tat.select_action(t_0)
    action_1 = tit_for_tat.select_action(t_1)
    assert jnp.array_equal(action_0, action_1)

    t_0, t_1 = env.step((action_0, action_1))
    assert jnp.array_equal(t_0.reward, t_1.reward)


def test_longer_game() -> None:
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    num_steps = 50
    num_inner_steps = 2
    env = IteratedMatrixGame(num_envs, payoff, num_inner_steps, num_steps)
    t_0, t_1 = env.reset()

    agent = TitForTat(num_envs)
    action = agent.select_action(t_0)

    r1 = []
    r2 = []
    for _ in range(10):
        action = agent.select_action(t_0)
        t0, t1 = env.step((action, action))
        r1.append(t0.reward)
        r2.append(t1.reward)

    assert jnp.array_equal(t_0.reward, t_1.reward)
    assert jnp.mean(jnp.stack(r1)) == 2
    assert jnp.mean(jnp.stack(r2)) == 2


def test_done():
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = IteratedMatrixGame(num_envs, payoff, 5, 5)
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

    # check back at start
    assert jnp.array_equal(t_0.observation.argmax(), 4)
    assert jnp.array_equal(t_1.observation.argmax(), 4)


def test_reset():
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    env = IteratedMatrixGame(num_envs, payoff, 5, 10)
    state = jnp.ones((num_envs,))

    env.reset()

    for _ in range(4):
        t_0, t_1 = env.step((0 * state, 0 * state))
        assert t_0.last().all() == False
        assert t_1.last().all() == False

    env.reset()

    for _ in range(4):
        t_0, t_1 = env.step((0 * state, 0 * state))
        assert t_0.last().all() == False
        assert t_1.last().all() == False
