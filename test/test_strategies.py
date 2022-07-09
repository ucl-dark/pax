import jax
import jax.numpy as jnp
from pax.env import InfiniteMatrixGame
from pax.strategies import MetaNaiveLearner, TitForTat, GrimTrigger
from dm_env import TimeStep, transition, termination


def test_titfortat():
    agent = TitForTat()
    batch_number = 3

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0]])
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

    # test initial conditions
    cd_timestep = transition(observation=initial_obs, reward=0)
    action = agent.select_action(cd_timestep)
    assert jnp.array_equal(cooperate_action, action)


def test_grim():
    agent = GrimTrigger()
    batch_number = 3

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0]])
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


def test_naive():
    batch_number = 1
    env = InfiniteMatrixGame(
        num_envs=batch_number,
        payoff=[[2, 2], [3, 0], [0, 3], [1, 1]],
        episode_length=jnp.inf,
        gamma=0.96,
    )
    agent = MetaNaiveLearner(
        action_dim=5, env=env, lr=1, seed=jax.random.PRNGKey(0)
    )

    alt_action = jnp.ones((batch_number, 5))

    timestep = env.reset()

    action = agent.select_action(timestep)
    assert action.shape == (batch_number, 5)

    for _ in range(5):
        action = agent.select_action(timestep)
        next_timestep = env.step([action, alt_action])
        print("Reward", next_timestep[0].reward)

        agent.update(timestep[0], action, next_timestep[0])
        timestep = next_timestep

    action = agent.select_action(timestep)

    assert env.step([action, alt_action])[0].reward == 3.0
