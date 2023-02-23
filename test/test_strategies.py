import jax
import jax.numpy as jnp

from pax.agents.naive_exact import NaiveExact
from pax.agents.strategies import EvilGreedy, GrimTrigger, TitForTat
from pax.envs.coin_game import CoinGame
from pax.envs.coin_game import EnvParams as CoinGameParams
from pax.envs.coin_game import EnvState as CoinGameState
from pax.envs.infinite_matrix_game import EnvParams as InfiniteMatrixGameParams
from pax.envs.infinite_matrix_game import InfiniteMatrixGame


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

    for _ in range(50):
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
    env.reset = jax.jit(
        jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    )
    env.step = jax.jit(
        jax.vmap(
            env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
        )
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
    defect_action = -20 * jnp.ones((num_envs, 5))

    for _ in range(50):
        rng, _ = split(rng, 2)
        action, agent_state, agent_memory = agent._policy(
            agent_state, obs1, agent_memory
        )
        (obs1, obs2), env_state, _, _, _ = env.step(
            rng, env_state, (action, defect_action), env_params
        )

    action, _, _ = agent._policy(agent_state, obs1, agent_memory)
    _, _, (reward0, reward1), _, _ = env.step(
        rng, env_state, (action, defect_action), env_params
    )
    assert jnp.allclose(0.99, reward0, atol=0.01)


def test_naive_tft():
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
    env.reset = jax.jit(
        jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    )
    env.step = jax.jit(
        jax.vmap(
            env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
        )
    )

    (obs1, _), env_state = env.reset(rng, env_params)
    agent = NaiveExact(
        action_dim=5,
        env_params=env_params,
        lr=1,
        num_envs=num_envs,
        player_id=0,
    )
    agent_state, agent_memory = agent.make_initial_state(obs1)
    tft_action = jnp.tile(
        20 * jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]), (num_envs, 1)
    )

    for _ in range(50):
        rng, _ = split(rng, 2)
        action, agent_state, agent_memory = agent._policy(
            agent_state, obs1, agent_memory
        )

        (obs1, _), env_state, rewards, _, _ = env.step(
            rng, env_state, (action, tft_action), env_params
        )

    action, _, _ = agent._policy(agent_state, obs1, agent_memory)
    _, _, (reward0, reward1), _, _ = env.step(
        rng, env_state, (action, tft_action), env_params
    )
    assert jnp.allclose(2.0, reward0, atol=0.01)

    (obs1, obs2), env_state = env.reset(rng, env_params)
    agent = NaiveExact(
        action_dim=5,
        env_params=env_params,
        lr=10,
        num_envs=num_envs,
        player_id=0,
    )
    agent_state, agent_memory = agent.make_initial_state(obs1)
    defect_action = -20 * jnp.ones((num_envs, 5))

    for _ in range(50):
        rng, _ = split(rng, 2)
        action, agent_state, agent_memory = agent._policy(
            agent_state, obs1, agent_memory
        )
        (obs1, obs2), env_state, _, _, _ = env.step(
            rng, env_state, (action, defect_action), env_params
        )

    action, _, _ = agent._policy(agent_state, obs1, agent_memory)
    _, _, (reward0, reward1), _, _ = env.step(
        rng, env_state, (action, defect_action), env_params
    )
    assert jnp.allclose(0.99, reward0, atol=0.01)


def test_naive_tft():
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
    env.reset = jax.jit(
        jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    )
    env.step = jax.jit(
        jax.vmap(
            env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
        )
    )

    (obs1, _), env_state = env.reset(rng, env_params)
    agent = NaiveExact(
        action_dim=5,
        env_params=env_params,
        lr=1,
        num_envs=num_envs,
        player_id=0,
    )
    agent_state, agent_memory = agent.make_initial_state(obs1)
    tft_action = jnp.tile(
        20 * jnp.array([1.0, -1.0, 1.0, -1.0, 1.0]), (num_envs, 1)
    )

    for _ in range(50):
        rng, _ = split(rng, 2)
        action, agent_state, agent_memory = agent._policy(
            agent_state, obs1, agent_memory
        )

        (obs1, _), env_state, rewards, _, _ = env.step(
            rng, env_state, (action, tft_action), env_params
        )

    action, _, _ = agent._policy(agent_state, obs1, agent_memory)
    _, _, (reward0, reward1), _, _ = env.step(
        rng, env_state, (action, tft_action), env_params
    )
    assert jnp.allclose(2.0, reward0, atol=0.01)


def test_naive_tft_as_second_player():
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
    env.reset = jax.jit(
        jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    )
    env.step = jax.jit(
        jax.vmap(
            env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
        )
    )

    (_, obs2), env_state = env.reset(rng, env_params)

    agent = NaiveExact(
        action_dim=5,
        env_params=env_params,
        lr=1,
        num_envs=num_envs,
        player_id=0,
    )
    agent_state, agent_memory = agent.make_initial_state(obs2)
    tft_action2 = jnp.tile(
        20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (num_envs, 1)
    )

    for _ in range(50):
        rng, _ = split(rng, 2)
        action, agent_state, agent_memory = agent._policy(
            agent_state, obs2, agent_memory
        )

        (_, obs2), env_state, rewards, _, _ = env.step(
            rng, env_state, (tft_action2, action), env_params
        )

    action, _, _ = agent._policy(agent_state, obs2, agent_memory)
    _, _, (_, reward1), _, _ = env.step(
        rng, env_state, (tft_action2, action), env_params
    )
    assert jnp.allclose(2.0, reward1, atol=0.01)


def test_coin_chaser():
    rng = jax.random.PRNGKey(0)
    env = CoinGame(
        num_inner_steps=8, num_outer_steps=2, cnn=True, egocentric=True
    )
    env_param = CoinGameParams(payoff_matrix=[[1, 1, -2], [1, 1, -2]])
    obs, state = env.reset(rng, env_param)

    env_state = CoinGameState(
        red_pos=jnp.array([0, 0]),
        blue_pos=jnp.array([2, 0]),
        red_coin_pos=jnp.array([0, 2]),
        blue_coin_pos=jnp.array([2, 1]),
        inner_t=state.inner_t,
        outer_t=state.outer_t,
        red_coop=0,
        red_defect=0,
        blue_coop=0,
        blue_defect=0,
        counter=state.counter,
        coop1=state.coop1,
        coop2=state.coop2,
        last_state=state.last_state,
    )

    #  r 0 rc
    #  0 0 0
    #  b bc 0
    # update locations (agent 1: [0, 0] agent 2: [2, 1])
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (4, 4), env_param
    )
    agent = EvilGreedy(4)
    a1 = agent._greedy_step(obs[0])
    # take a left
    assert a1 == 1

    # take a right
    a2 = agent._greedy_step(obs[1])
    assert a2 == 0
