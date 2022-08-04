import jax
import jax.numpy as jnp
from pax.meta_env import InfiniteMatrixGame
from pax.strategies import TitForTat, GrimTrigger
from pax.naive_exact import NaiveLearnerEx
from dm_env import transition


def test_titfortat():
    agent = TitForTat()
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
    agent = GrimTrigger()
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
    batch_number = 1
    env = InfiniteMatrixGame(
        num_envs=batch_number,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveLearnerEx(action_dim=5, env=env, lr=10, seed=0, player_id=0)

    alt_action = 20 * jnp.ones((batch_number, 5))
    timestep, _ = env.reset()

    for _ in range(500):
        action = agent.select_action(timestep)
        print(jax.nn.sigmoid(action))
        next_timestep, _ = env.step([action, alt_action])
        agent.update(timestep, action, next_timestep)
        timestep = next_timestep

    action = agent.select_action(timestep)
    assert jnp.allclose(
        3.0, env.step([action, alt_action])[0].reward, atol=0.01
    )


def test_naive_defect():
    batch_number = 5
    env = InfiniteMatrixGame(
        num_envs=batch_number,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveLearnerEx(action_dim=5, env=env, lr=1, seed=0, player_id=0)

    defect_action = -20 * jnp.ones((batch_number, 5))
    timestep, _ = env.reset()

    for _ in range(50):
        action = agent.select_action(timestep)
        next_timestep, _ = env.step([action, defect_action])
        agent.update(timestep, action, next_timestep)
        timestep = next_timestep

    action = agent.select_action(timestep)
    # assert jnp.allclose(action, jnp.zeros((batch_number, 5)))
    assert jnp.allclose(
        0.99, env.step([action, defect_action])[0].reward, atol=0.01
    )


def test_naive_tft():
    batch_number = 1
    env = InfiniteMatrixGame(
        num_envs=batch_number,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveLearnerEx(action_dim=5, env=env, lr=1, seed=0, player_id=0)
    tft_action = jnp.tile(
        20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (batch_number, 1)
    )
    timestep, _ = env.reset()

    for _ in range(50):
        action = agent.select_action(timestep)
        next_timestep, _ = env.step([action, tft_action])
        agent.update(timestep, action, next_timestep)
        timestep = next_timestep

    action = agent.select_action(timestep)
    assert jnp.allclose(
        env.step([action, tft_action])[0].reward, 2.0, atol=0.01
    )


def test_naive_tft_as_second_player():
    batch_number = 1
    env = InfiniteMatrixGame(
        num_envs=batch_number,
        payoff=[[2, 2], [0, 3], [3, 0], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
        seed=0,
    )
    agent = NaiveLearnerEx(action_dim=5, env=env, lr=1, seed=0, player_id=0)

    tft_action = jnp.tile(
        20 * jnp.array([[1.0, -1.0, 1.0, -1.0, 1.0]]), (batch_number, 1)
    )
    _, timestep = env.reset()

    for _ in range(50):
        action = agent.select_action(timestep)
        _, next_timestep = env.step([tft_action, action])
        timestep = next_timestep

    assert jnp.allclose(
        env.step([tft_action, action])[1].reward, 2.0, atol=0.01
    )
