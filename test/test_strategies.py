import jax.numpy as jnp
import jax
from pax.envs.infinite_matrix_game import (
    InfiniteMatrixGame,
    EnvParams as InfiniteMatrixGameParams,
)
from pax.envs.coin_game import CoinGame, EnvParams as CoinGameState
from pax.strategies import EvilGreedy, TitForTat, GrimTrigger
from pax.naive_exact import NaiveExact


def test_titfortat():
    batch_number = 3
    agent = TitForTat(num_envs=1)

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0]])
    dd_obs = jnp.asarray(batch_number * [[0, 0, 0, 1, 0]])
    initial_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 1]])

    cooperate_action = 0 * jnp.ones((batch_number,), dtype=jnp.int32)
    defect_action = 1 * jnp.ones((batch_number,), dtype=jnp.int32)

    action, _, _ = agent._policy(None, cc_obs, None)
    assert jnp.array_equal(
        cooperate_action, action
    )  # , f"cooperate_action = {cooperate_action}, action = {action}"

    action, _, _ = agent._policy(None, dd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, dc_obs, None)
    # expect first player to cooperate, expect second player to defect
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, cd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, initial_obs, None)
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

    action, _, _ = agent._policy(None, cc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, cd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, dc_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, dd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, initial_obs, None)
    assert jnp.array_equal(cooperate_action, action)


def test_naive_alt():
    num_envs = 2

    rng = jnp.concatenate([jax.random.PRNGKey(0)] * num_envs).reshape(
        num_envs, -1
    )

    env = InfiniteMatrixGame(num_steps=jnp.inf)
    env_params = InfiniteMatrixGameParams(
        payoff_matrix=[[2, 2], [0, 3], [3, 0], [1, 1]], gamma=0.96
    )

    # vmaps
    split = jax.vmap(jax.random.split, in_axes=(0, None))
    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    (obs1, obs2), env_state = env.reset(rng, env_params)

    agent = NaiveExact(
        action_dim=5,
        env_params=env_params,
        lr=10,
        num_envs=num_envs,
        player_id=0,
    )
    agent_state, agent_memory = agent.make_initial_state(obs1)

    alt_action = 20 * jnp.ones((num_envs, 5))

    for _ in range(500):
        rng, _ = split(rng, 2)
        action, agent_state, agent_memory = agent._policy(
            agent_state, obs1, agent_memory
        )
        (obs1, obs2), env_state, _, _, _ = env.step(
            rng, env_state, (action, alt_action), env_params
        )

    action, _, _ = agent._policy(agent_state, obs1, agent_memory)
    _, _, (reward0, reward1), _, _ = env.step(
        rng, env_state, (action, alt_action), env_params
    )
    assert jnp.allclose(3.0, reward0, atol=0.01)


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
    env = InfiniteMatrixGame(num_steps=jnp.inf)
    # env_params = InfiniteMatrixGameParams()

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
