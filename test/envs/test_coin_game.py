import jax.numpy as jnp
import jax

from pax.envs.coin_game import CoinGame, EnvParams, EnvState


def test_coingame_shapes():
    rng = jax.random.PRNGKey(0)
    env = CoinGame(
        num_inner_steps=8, num_outer_steps=2, cnn=False, egocentric=True
    )

    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])

    obs, state = env.reset(rng, params)
    action = jnp.ones((), dtype=int)

    assert obs[0].shape == (36,)
    assert obs[1].shape == (36,)

    obs, new_state, rewards, done, info = env.step(
        rng, state, (action, action), params
    )
    assert rewards[0].shape == ()
    assert rewards[1].shape == ()


def test_coingame_move():
    rng = jax.random.PRNGKey(0)
    env = CoinGame(8, 2, True, True)
    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])

    action = 1
    _, state = env.reset(rng, params)

    state = EnvState(
        red_pos=jnp.array([0, 0]),
        blue_pos=jnp.array([1, 0]),
        red_coin_pos=jnp.array([0, 2]),
        blue_coin_pos=jnp.array([1, 2]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )

    _, new_state, rewards, _, _ = env.step(
        rng, state, (action, action), params
    )
    assert (state.red_pos != new_state.red_pos).any()
    assert (state.blue_pos != new_state.blue_pos).any()
    state = new_state
    assert rewards[0] == 1
    assert rewards[1] == 1
    assert (state.red_coop == jnp.array([1, 0])).all()
    assert (state.red_defect == jnp.array([0, 0])).all()
    assert (state.blue_coop == jnp.array([1, 0])).all()
    assert (state.blue_defect == jnp.array([0, 0])).all()
    assert state.inner_t == 1
    assert state.outer_t == 0

    obs, state, rewards, done, info = env.step(
        rng, state, (action, action), params
    )
    assert rewards[0] == 0
    assert rewards[1] == 0
    assert (state.red_coop == jnp.array([1, 0])).all()
    assert (state.red_defect == jnp.array([0, 0])).all()
    assert (state.blue_coop == jnp.array([1, 0])).all()
    assert (state.blue_defect == jnp.array([0, 0])).all()
    assert state.inner_t == 2
    assert state.outer_t == 0


def test_coingame_egocentric_colors():
    rng = jax.random.PRNGKey(0)
    env = CoinGame(8, 2, True, True)
    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    _, state = env.reset(rng, params)
    action = 1

    # importantly place agents on top of each other so that we can just check flips
    state = EnvState(
        red_pos=jnp.array([0, 0]),
        blue_pos=jnp.array([0, 0]),
        red_coin_pos=jnp.array([0, 2]),
        blue_coin_pos=jnp.array([0, 2]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )

    for _ in range(7):
        obs, state, rewards, done, info = env.step(
            rng, state, (action, action), params
        )
        obs1, obs2 = obs[0], obs[1]
        # remove batch
        assert (obs1[:, :, 0] == obs2[:, :, 1]).all()
        assert (obs1[:, :, 1] == obs2[:, :, 0]).all()
        assert (obs1[:, :, 2] == obs2[:, :, 3]).all()
        assert (obs1[:, :, 3] == obs2[:, :, 2]).all()

    # here we reset the environment so expect these to break
    obs, state, rewards, done, info = env.step(
        rng, state, (action, action), params
    )
    obs1, obs2 = obs[0], obs[1]
    assert (obs1[:, :, 0] != obs2[:, :, 1]).any()
    assert (obs1[:, :, 1] != obs2[:, :, 0]).any()
    assert (obs1[:, :, 2] != obs2[:, :, 3]).any()
    assert (obs1[:, :, 3] != obs2[:, :, 2]).any()


def test_coingame_egocentric_pos():
    rng = jax.random.PRNGKey(0)
    env = CoinGame(8, 2, True, True)
    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    _, state = env.reset(rng, params)
    action = 1

    state = EnvState(
        red_pos=jnp.array([1, 2]),
        blue_pos=jnp.array([2, 2]),
        red_coin_pos=jnp.array([2, 2]),
        blue_coin_pos=jnp.array([0, 0]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )
    # it would be nice to have a stay action here lol
    obs, state, rewards, done, info = env.step(
        rng, state, (action, action), params
    )

    # takes left so red_pos = [1, 1], blue_pos = [2, 1]
    obs1, obs2 = obs[0], obs[1]

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


def test_coingame_stay():
    rng = jax.random.PRNGKey(0)
    env = CoinGame(8, 2, True, True)
    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])

    action = 4
    _, state = env.reset(rng, params)

    _state = EnvState(
        red_pos=jnp.array([0, 0]),
        blue_pos=jnp.array([1, 0]),
        red_coin_pos=jnp.array([0, 2]),
        blue_coin_pos=jnp.array([1, 2]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )

    obs, state, rewards, done, info = env.step(
        rng, _state, (action, action), params
    )
    assert (state.red_pos == _state.red_pos).all()
    assert (state.blue_pos == _state.blue_pos).all()
    assert state.inner_t != _state.inner_t


def test_coingame_batch_egocentric():
    bs = 1
    action = jnp.ones(bs, dtype=int)
    rngs = jnp.concatenate(bs * [jax.random.PRNGKey(0)]).reshape((bs, -1))
    env = CoinGame(8, 2, True, True)
    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])

    env.reset = jax.vmap(env.reset, in_axes=(0, None))
    env.step = jax.vmap(
        env.step,
        in_axes=(0, 0, 0, None),
    )

    obs, state = env.reset(rngs, params)
    state = EnvState(
        red_pos=jnp.array([[0, 0]]),
        blue_pos=jnp.array([[0, 0]]),
        red_coin_pos=jnp.array([[0, 2]]),
        blue_coin_pos=jnp.array([[0, 2]]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )
    obs1, obs2 = obs[0], obs[1]

    for _ in range(7):
        obs, state, rewards, done, info = env.step(
            rngs, state, (5 * action, 5 * action), params
        )
        obs1, obs2 = obs[0][0], obs[1][0]
        # remove batch
        assert (obs1[:, :, 0] == obs2[:, :, 1]).all()
        assert (obs1[:, :, 1] == obs2[:, :, 0]).all()
        assert (obs1[:, :, 2] == obs2[:, :, 3]).all()
        assert (obs1[:, :, 3] == obs2[:, :, 2]).all()

    # check negative case
    env.state = EnvState(
        red_pos=jnp.array([[0, 0]]),
        blue_pos=jnp.array([[2, 2]]),
        red_coin_pos=jnp.array([[0, 1]]),
        blue_coin_pos=jnp.array([[0, 1]]),
        inner_t=0 * state.inner_t,
        outer_t=0 * state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )

    obs, state, rewards, done, info = env.step(
        rngs, state, (5 * action, 5 * action), params
    )
    obs1, obs2 = obs[0][0], obs[1][0]

    assert (obs1[:, :, 0] != obs2[:, :, 1]).any()
    assert (obs1[:, :, 1] != obs2[:, :, 0]).any()
    assert (obs1[:, :, 2] != obs2[:, :, 3]).any()
    assert (obs1[:, :, 3] != obs2[:, :, 2]).any()


def test_coingame_non_egocentric_batched():
    bs = 1
    action = jnp.ones(bs, dtype=int)
    rngs = jnp.concatenate(bs * [jax.random.PRNGKey(0)]).reshape((bs, -1))
    env = CoinGame(8, 2, True, False)
    params = EnvParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])

    env.reset = jax.vmap(env.reset, in_axes=(0, None))
    env.step = jax.vmap(
        env.step,
        in_axes=(0, 0, 0, None),
    )

    _, state = env.reset(rngs, params)
    state = EnvState(
        red_pos=jnp.array([[0, 0]]),
        blue_pos=jnp.array([[2, 2]]),
        red_coin_pos=jnp.array([[1, 1]]),
        blue_coin_pos=jnp.array([[1, 1]]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )

    obs, state, rewards, done, info = env.step(
        rngs, state, (5 * action, 5 * action), params
    )
    obs1, obs2 = obs[0][0], obs[1][0]

    expected_obs1 = jnp.array(
        [
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # agent
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # other agent
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # agent coin
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # other coin
        ],
        dtype=jnp.int8,
    )
    # channel last
    expected_obs1 = jnp.transpose(expected_obs1, (1, 2, 0))

    assert (obs1[:, :, 0] == expected_obs1[:, :, 0]).all()
    assert (expected_obs1 == obs1).all()

    expected_obs2 = jnp.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 1]],  # agent
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],  # other agent
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # agent coin
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],  # other coin
        ],
        dtype=jnp.int8,
    )
    expected_obs2 = jnp.transpose(expected_obs2, (1, 2, 0))
    assert (expected_obs2 == obs2).all()

    for _ in range(6):
        obs, state, rewards, done, info = env.step(
            rngs, state, (action, action), params
        )
        obs1, obs2 = obs[0][0], obs[1][0]
        # remove batch
        assert (obs1[:, :, 0] == obs2[:, :, 1]).all()
        assert (obs1[:, :, 1] == obs2[:, :, 0]).all()
        assert (obs1[:, :, 2] == obs2[:, :, 3]).all()
        assert (obs1[:, :, 3] == obs2[:, :, 2]).all()
