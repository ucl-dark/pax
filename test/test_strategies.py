import jax.numpy as jnp
import numpy as np
from jax import vmap

from src.strategies import TitForTat


def test_titfortat():
    agent = TitForTat()
    batch_number = 3

    # all obs are only of final state e.g batch x dim.
    cc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0, 0]])
    dc_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0, 0]])
    cd_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0, 0]])
    dd_obs = jnp.asarray(batch_number * [[0, 0, 0, 1, 0, 0]])
    initial_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 1, 0]])

    cooperate_action = 0 * jnp.ones((batch_number, 1))
    defect_action = 1 * jnp.ones((batch_number, 1))

    action, _ = agent.actor_step(None, cc_obs)
    assert jnp.array_equal(cooperate_action, action)

    action, _ = agent.actor_step(None, dd_obs)
    assert jnp.array_equal(defect_action, action)

    action, _ = agent.actor_step(None, dc_obs)
    # expect first player to cooperate, expect second player to defect
    assert jnp.array_equal(cooperate_action, action)

    action, _ = agent.actor_step(None, cd_obs)
    assert jnp.array_equal(defect_action, action)

    # test initial conditions
    action, _ = agent.actor_step(None, initial_obs)
    assert jnp.array_equal(cooperate_action, action)
