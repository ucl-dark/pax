import jax.numpy as jnp
import jax
from pax.strategies import TitForTat
from dm_env import TimeStep, transition, termination


def test_titfortat():
    agent = TitForTat()
    batch_number = 3
    dummy_key = jax.random.PRNGKey(0)

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0, 0]])
    dd_obs = jnp.asarray(batch_number * [[0, 0, 0, 1, 0, 0]])
    initial_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 1, 0]])

    cooperate_action = 0 * jnp.ones((batch_number, 1))
    defect_action = 1 * jnp.ones((batch_number, 1))

    cc_timestep = transition(observation=cc_obs, reward=0)
    action = agent.select_action(dummy_key, cc_timestep)
    assert jnp.array_equal(cooperate_action, action)

    dd_timestep = transition(observation=dd_obs, reward=0)
    action = agent.select_action(dummy_key, dd_timestep)
    assert jnp.array_equal(defect_action, action)

    dc_timestep = transition(observation=dc_obs, reward=0)
    action = agent.select_action(dummy_key, dc_timestep)
    # expect first player to cooperate, expect second player to defect
    assert jnp.array_equal(cooperate_action, action)

    cd_timestep = transition(observation=cd_obs, reward=0)
    action = agent.select_action(dummy_key, cd_timestep)
    assert jnp.array_equal(defect_action, action)

    # test initial conditions
    cd_timestep = transition(observation=initial_obs, reward=0)
    action = agent.select_action(dummy_key, cd_timestep)
    assert jnp.array_equal(cooperate_action, action)
