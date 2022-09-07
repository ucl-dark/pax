from multiprocessing.spawn import old_main_modules
import jax.numpy as jnp
import pytest

from pax.env_meta import CoinGame, CoinGameState, MetaFiniteGame
from pax.strategies import TitForTat

from pax.env_inner import InfiniteMatrixGame

# payoff matrices for four games
ipd = [[2, 2], [0, 3], [3, 0], [1, 1]]
stag = [[4, 4], [1, 3], [3, 1], [2, 2]]
sexes = [[3, 2], [0, 0], [0, 0], [2, 3]]
chicken = [[0, 0], [-1, 1], [1, -1], [-2, -2]]
test_payoffs = [ipd, stag, sexes, chicken]


@pytest.mark.parametrize("payoff", test_payoffs)
def test_single_batch_rewards(payoff) -> None:
    num_envs = 1
    env = MetaFiniteGame(num_envs, payoff, 5, 10)
    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)

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
    env = MetaFiniteGame(num_envs, payoff, 5, 10)
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
    env = MetaFiniteGame(num_envs, payoff, 5, 10)
    t_0, t_1 = env.reset()

    tit_for_tat = TitForTat()

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
    env = MetaFiniteGame(num_envs, payoff, num_inner_steps, num_steps)
    t_0, t_1 = env.reset()

    agent = TitForTat()
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
    env = MetaFiniteGame(num_envs, payoff, 5, 5)
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
    env = MetaFiniteGame(num_envs, payoff, 5, 10)
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


def test_coingame_shapes():
    batch_size = 2
    env = CoinGame(batch_size, 8, 16, 0, False)
    action = jnp.ones(batch_size, dtype=int)

    t1, t2 = env.reset()
    assert (t1.reward == jnp.zeros(batch_size)).all()
    assert (t2.reward == jnp.zeros(batch_size)).all()
    old_state = env.state

    assert t1.observation.shape == (batch_size, 36)
    assert t2.observation.shape == (batch_size, 36)

    t1, t2 = env.step((action, action))
    assert (t1.reward == jnp.zeros(batch_size)).all()
    assert (t2.reward == jnp.zeros(batch_size)).all()
    new_state = env.state

    assert (old_state.red_pos != new_state.red_pos).any()
    assert (old_state.blue_pos != new_state.blue_pos).any()
    assert (old_state.key != new_state.key).all()


def test_coingame_move():
    bs = 1
    env = CoinGame(bs, 8, 16, 0, True)
    action = jnp.ones(bs, dtype=int)
    t1, t2 = env.reset()
    env.state = CoinGameState(
        red_pos=jnp.array([[0, 0]]),
        blue_pos=jnp.array([[1, 0]]),
        red_coin_pos=jnp.array([[0, 2]]),
        blue_coin_pos=jnp.array([[1, 2]]),
        key=env.state.key,
        inner_t=env.state.inner_t,
        outer_t=env.state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
    )

    t1, t2 = env.step((action, action))
    assert t1.reward == 1
    assert t2.reward == 1
    assert env.state.red_coop == 1
    assert env.state.red_defect == 0
    assert env.state.blue_coop == 1
    assert env.state.blue_defect == 0

    t1, t2 = env.step((action, action))
    assert t1.reward == 0
    assert t2.reward == 0
    assert env.state.red_coop == 1
    assert env.state.red_defect == 0
    assert env.state.blue_coop == 1
    assert env.state.blue_defect == 0


def test_coingame_egocentric_colors():
    bs = 1
    env = CoinGame(bs, 8, 16, 0, True)
    action = jnp.ones(bs, dtype=int)
    t1, t2 = env.reset()

    # importantly place agents on top of each other so that we can just check flips
    env.state = CoinGameState(
        red_pos=jnp.array([[0, 0]]),
        blue_pos=jnp.array([[0, 0]]),
        red_coin_pos=jnp.array([[0, 2]]),
        blue_coin_pos=jnp.array([[0, 2]]),
        key=env.state.key,
        inner_t=env.state.inner_t,
        outer_t=env.state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
    )

    for _ in range(7):
        t1, t2 = env.step((action, action))
        obs1, obs2 = t1.observation[0], t2.observation[0]
        # remove batch
        assert (obs1[:, :, 0] == obs2[:, :, 1]).all()
        assert (obs1[:, :, 1] == obs2[:, :, 0]).all()
        assert (obs1[:, :, 2] == obs2[:, :, 3]).all()
        assert (obs1[:, :, 3] == obs2[:, :, 2]).all()

    # here we reset the environment so expect these to break
    t1, t2 = env.step((action, action))
    obs1, obs2 = t1.observation[0], t2.observation[0]
    assert (obs1[:, :, 0] != obs2[:, :, 1]).any()
    assert (obs1[:, :, 1] != obs2[:, :, 0]).any()
    assert (obs1[:, :, 2] != obs2[:, :, 3]).any()
    assert (obs1[:, :, 3] != obs2[:, :, 2]).any()


def test_coingame_egocentric_pos():
    bs = 1
    env = CoinGame(bs, 8, 16, 0, True)
    action = jnp.ones(bs, dtype=int)
    t1, t2 = env.reset()

    env.state = CoinGameState(
        red_pos=jnp.array([[1, 2]]),
        blue_pos=jnp.array([[2, 2]]),
        red_coin_pos=jnp.array([[2, 2]]),
        blue_coin_pos=jnp.array([[0, 0]]),
        key=env.state.key,
        inner_t=env.state.inner_t,
        outer_t=env.state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
    )
    # it would be nice to have a stay action here lol
    t1, t2 = env.step((action, action))

    # takes left so red_pos = [1, 1], blue_pos = [2, 1]

    obs1, obs2 = t1.observation[0], t2.observation[0]

    expected_obs1 = jnp.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # agent
            [[0, 0, 0], [0, 0, 0], [0, 1, 0]],  # other agent
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # agent coin
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # other coin
        ],
        dtype=jnp.int8,
    )
    # channel last
    expected_obs1 = jnp.transpose(expected_obs1, (1, 2, 0))
    assert (expected_obs1 == obs1).all()

    expected_obs2 = jnp.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # agent
            [[0, 1, 0], [0, 0, 0], [0, 0, 0]],  # other agent
            [[0, 0, 0], [0, 0, 0], [1, 0, 0]],  # agent coin
            [[0, 0, 0], [0, 0, 1], [0, 0, 0]],  # other coin
        ],
        dtype=jnp.int8,
    )
    expected_obs2 = jnp.transpose(expected_obs2, (1, 2, 0))
    assert (expected_obs2 == obs2).all()
