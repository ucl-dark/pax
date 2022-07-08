import jax.numpy as jnp
from pax.strategies import TitForTat, GrimTrigger
from dm_env import TimeStep, transition, termination


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
    assert jnp.array_equal(cooperate_action, action)

    cd_timestep = transition(observation=cd_obs, reward=0)
    action = agent.select_action(cd_timestep)
    assert jnp.array_equal(defect_action, action)

    dc_timestep = transition(observation=dc_obs, reward=0)
    action = agent.select_action(dc_timestep)
    assert jnp.array_equal(cooperate_action, action)

    dd_timestep = transition(observation=dd_obs, reward=0)
    action = agent.select_action(dd_timestep)
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
