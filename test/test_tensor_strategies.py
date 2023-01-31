import jax
import jax.numpy as jnp

from pax.agents.tensor_strategies import (
    TitForTatStrictStay,
    TitForTatStrictSwitch,
)

batch_number = 3
# all obs are only of final state e.g batch x dim.
ccc_obs = jnp.asarray(batch_number * [[1, 0, 0, 0, 0, 0, 0, 0, 0]])
ccd_obs = jnp.asarray(batch_number * [[0, 1, 0, 0, 0, 0, 0, 0, 0]])
cdc_obs = jnp.asarray(batch_number * [[0, 0, 1, 0, 0, 0, 0, 0, 0]])
cdd_obs = jnp.asarray(batch_number * [[0, 0, 0, 1, 0, 0, 0, 0, 0]])
dcc_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 1, 0, 0, 0, 0]])
dcd_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 0, 1, 0, 0, 0]])
ddc_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 0, 0, 1, 0, 0]])
ddd_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 0, 0, 0, 1, 0]])
initial_obs = jnp.asarray(batch_number * [[0, 0, 0, 0, 0, 0, 0, 0, 1]])

cooperate_action = 0 * jnp.ones((batch_number,), dtype=jnp.int32)
defect_action = 1 * jnp.ones((batch_number,), dtype=jnp.int32)


def test_titfortat_strict_stay():
    # Switch to what opponents did if the other two played the same move
    # otherwise play as before
    agent = TitForTatStrictStay(num_envs=1)
    action, _, _ = agent._policy(None, ccc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, ccd_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, cdc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, cdd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, dcc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, dcd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, ddc_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, ddd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, initial_obs, None)
    assert jnp.array_equal(cooperate_action, action)


def test_titfortat_strict_switch():
    # Play what opponents did if the other two played the same move
    # otherwise switch from previous
    agent = TitForTatStrictSwitch(num_envs=1)

    action, _, _ = agent._policy(None, ccc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, ccd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, cdc_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, cdd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, dcc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, dcd_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, ddc_obs, None)
    assert jnp.array_equal(cooperate_action, action)

    action, _, _ = agent._policy(None, ddd_obs, None)
    assert jnp.array_equal(defect_action, action)

    action, _, _ = agent._policy(None, initial_obs, None)
    assert jnp.array_equal(cooperate_action, action)
