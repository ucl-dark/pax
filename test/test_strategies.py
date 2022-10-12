import jax
import jax.numpy as jnp
from pax.env_inner import InfiniteMatrixGame
from pax.env_meta import CoinGame, CoinGameState
from pax.strategies import EvilGreedy, TitForTat, GrimTrigger
from pax.naive_exact import NaiveExact
from dm_env import transition


def test_titfortat():
    agent = TitForTat(num_envs=1)
    batch_number = 3

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0]])
    dd_obs = jnp.asarray(batch_number * [[0, 0, 0, 1, 0]])
    initial_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 1]])

    cooperate_action = 0 * jnp.ones((batch_number,))
    defect_action = 1 * jnp.ones((batch_number,))

    cc_timestep = transition(observation=cc_obs, reward=0)
    action = agent.select_action(cc_timestep)
    assert jnp.array_equal(
        cooperate_action, action
    )  # , f"cooperate_action = {cooperate_action}, action = {action}"

    dd_timestep = transition(observation=dd_obs, reward=0)
    action = agent.select_action(dd_timestep)
    assert jnp.array_equal(defect_action, action)

    dc_timestep = transition(observation=dc_obs, reward=0)
    action = agent.select_action(dc_timestep)
    # expect first player to cooperate, expect second player to defect
    assert jnp.array_equal(cooperate_action, action)

    cd_timestep = transition(observation=cd_obs, reward=0)
    action = agent.select_action(cd_timestep)
    assert jnp.array_equal(defect_action, action)

    start_timestep = transition(observation=initial_obs, reward=0)
    action = agent.select_action(start_timestep)
    assert jnp.array_equal(cooperate_action, action)


def test_grim():
    agent = GrimTrigger(num_envs=1)
    batch_number = 3

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0]])
    dd_obs = jnp.asarray(batch_number * [[0, 0, 0, 1, 0]])
    initial_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 1]])

    cooperate_action = 0 * jnp.ones((batch_number,))
    defect_action = 1 * jnp.ones((batch_number,))

    cc_timestep = transition(observation=cc_obs, reward=0)
    action = agent.select_action(cc_timestep)
    assert jnp.array_equal(cooperate_action, action)

    cd_timestep = transition(observation=cd_obs, reward=0)
    action = agent.select_action(cd_timestep)
    assert jnp.array_equal(defect_action, action)

    dc_timestep = transition(observation=dc_obs, reward=0)
    action = agent.select_action(dc_timestep)
    assert jnp.array_equal(defect_action, action)

    dd_timestep = transition(observation=dd_obs, reward=0)
    action = agent.select_action(dd_timestep)
    assert jnp.array_equal(defect_action, action)

    start_timestep = transition(observation=initial_obs, reward=0)
    action = agent.select_action(start_timestep)
    assert jnp.array_equal(cooperate_action, action)


def test_naive_alt():
    bs = 1
    env = InfiniteMatrixGame(
        num_envs=bs,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveExact(action_dim=5, env=env, lr=10, num_envs=bs, player_id=0)

    alt_action = 20 * jnp.ones((bs, 5))
    timestep, _ = env.reset()

    for _ in range(500):
        action = agent.select_action(timestep)
        next_timestep, _ = env.step([action, alt_action])
        timestep = next_timestep

    action = agent.select_action(timestep)
    assert jnp.allclose(
        3.0, env.step([action, alt_action])[0].reward, atol=0.01
    )


def test_naive_defect():
    bs = 5
    env = InfiniteMatrixGame(
        num_envs=bs,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveExact(action_dim=5, env=env, lr=1, num_envs=bs, player_id=0)

    defect_action = -20 * jnp.ones((bs, 5))
    timestep, _ = env.reset()

    for _ in range(50):
        action = agent.select_action(timestep)
        next_timestep, _ = env.step([action, defect_action])
        timestep = next_timestep

    action = agent.select_action(timestep)
    # assert jnp.allclose(action, jnp.zeros((batch_number, 5)))
    assert jnp.allclose(
        0.99, env.step([action, defect_action])[0].reward, atol=0.01
    )


def test_naive_tft():
    bs = 1
    env = InfiniteMatrixGame(
        num_envs=bs,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveExact(action_dim=5, env=env, lr=1, num_envs=bs, player_id=0)
    tft_action = jnp.tile(
        20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (bs, 1)
    )
    timestep, _ = env.reset()

    for _ in range(50):
        action = agent.select_action(timestep)
        next_timestep, _ = env.step([action, tft_action])
        timestep = next_timestep

    action = agent.select_action(timestep)
    assert jnp.allclose(
        env.step([action, tft_action])[0].reward, 2.0, atol=0.01
    )


def test_naive_tft_as_second_player():
    bs = 1
    env = InfiniteMatrixGame(
        num_envs=bs,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveExact(action_dim=5, env=env, lr=1, num_envs=bs, player_id=0)

    tft_action = jnp.tile(
        20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (bs, 1)
    )
    _, timestep = env.reset()

    for _ in range(50):
        action = agent.select_action(timestep)
        _, next_timestep = env.step([tft_action, action])
        timestep = next_timestep

    assert jnp.allclose(
        env.step([tft_action, action])[1].reward, 2.0, atol=0.01
    )


def test_coin_chaser():
    bs = 1
    env = CoinGame(bs, 8, 16, 0, True, True)
    t1, t2 = env.reset()
    env.state = CoinGameState(
        red_pos=jnp.array([[0, 0]]),
        blue_pos=jnp.array([[2, 0]]),
        red_coin_pos=jnp.array([[0, 2]]),
        blue_coin_pos=jnp.array([[2, 1]]),
        key=env.state.key,
        inner_t=env.state.inner_t,
        outer_t=env.state.outer_t,
        red_coop=jnp.zeros(1),
        red_defect=jnp.zeros(1),
        blue_coop=jnp.zeros(1),
        blue_defect=jnp.zeros(1),
        counter=env.state.counter,
        coop1=env.state.coop1,
        coop2=env.state.coop2,
        last_state=env.state.last_state,
    )

    #  r 0 rc
    #  0 0 0
    #  b bc 0
    # update locations (agent 1: [0, 0] agent 2: [2, 1])
    t1, t2 = env.step(
        (4 * jnp.ones(bs, dtype=jnp.int8), 4 * jnp.ones(bs, dtype=jnp.int8))
    )

    agent = EvilGreedy(4)
    a1 = agent._greedy_step(t1.observation[0])
    # take a left
    assert a1 == 1

    print("akbir")
    # take a right
    a2 = agent._greedy_step(t2.observation[0])
    assert a2 == 0
