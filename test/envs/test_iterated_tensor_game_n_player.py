import jax
import jax.numpy as jnp
import pytest

from pax.agents.strategies import TitForTat
from pax.agents.tensor_strategies import TitForTatStrictStay
from pax.envs.iterated_tensor_game_n_player import (
    EnvParams,
    IteratedTensorGameNPlayer,
)

# 2 PLAYERS
payoff_table_2pl = [
    [4, jnp.nan],
    [2, 5],
    [jnp.nan, 3],
]
cc_p1, cc_p2 = payoff_table_2pl[0][0], payoff_table_2pl[0][0]
cd_p1, cd_p2 = payoff_table_2pl[1][0], payoff_table_2pl[1][1]
dc_p1, dc_p2 = payoff_table_2pl[1][1], payoff_table_2pl[1][0]
dd_p1, dd_p2 = payoff_table_2pl[2][1], payoff_table_2pl[2][1]
# states
cc_obs = 0
cd_obs = 1
dc_obs = 2
dd_obs = 3

# 3 PLAYERS
payoff_table_3pl = [
    [4, jnp.nan],
    [2.66, 5.66],
    [1.33, 4.33],
    [jnp.nan, 3],
]
# this is for verification
ccc_p1, ccc_p2, ccc_p3 = (
    payoff_table_3pl[0][0],
    payoff_table_3pl[0][0],
    payoff_table_3pl[0][0],
)
ccd_p1, ccd_p2, ccd_p3 = (
    payoff_table_3pl[1][0],
    payoff_table_3pl[1][0],
    payoff_table_3pl[1][1],
)
cdc_p1, cdc_p2, cdc_p3 = (
    payoff_table_3pl[1][0],
    payoff_table_3pl[1][1],
    payoff_table_3pl[1][0],
)
cdd_p1, cdd_p2, cdd_p3 = (
    payoff_table_3pl[2][0],
    payoff_table_3pl[2][1],
    payoff_table_3pl[2][1],
)
dcc_p1, dcc_p2, dcc_p3 = (
    payoff_table_3pl[1][1],
    payoff_table_3pl[1][0],
    payoff_table_3pl[1][0],
)
dcd_p1, dcd_p2, dcd_p3 = (
    payoff_table_3pl[2][1],
    payoff_table_3pl[2][0],
    payoff_table_3pl[2][1],
)
ddc_p1, ddc_p2, ddc_p3 = (
    payoff_table_3pl[2][1],
    payoff_table_3pl[2][1],
    payoff_table_3pl[2][0],
)
ddd_p1, ddd_p2, ddd_p3 = (
    payoff_table_3pl[3][1],
    payoff_table_3pl[3][1],
    payoff_table_3pl[3][1],
)

# states
ccc_obs = 0
ccd_obs = 1
cdc_obs = 2
cdd_obs = 3
dcc_obs = 4
dcd_obs = 5
ddc_obs = 6
ddd_obs = 7

# 4 PLAYERS
payoff_table_4pl = [
    [4, jnp.nan],
    [3, 6],
    [2, 5],
    [1, 4],
    [jnp.nan, 3],
]

cccc_p1, cccc_p2, cccc_p3, cccc_p4 = (
    payoff_table_4pl[0][0],
    payoff_table_4pl[0][0],
    payoff_table_4pl[0][0],
    payoff_table_4pl[0][0],
)
cccd_p1, cccd_p2, cccd_p3, cccd_p4 = (
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][1],
)
ccdc_p1, ccdc_p2, ccdc_p3, ccdc_p4 = (
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][1],
    payoff_table_4pl[1][0],
)
ccdd_p1, ccdd_p2, ccdd_p3, ccdd_p4 = (
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][1],
)
cdcc_p1, cdcc_p2, cdcc_p3, cdcc_p4 = (
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][1],
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][0],
)
cdcd_p1, cdcd_p2, cdcd_p3, cdcd_p4 = (
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][1],
)
cddc_p1, cddc_p2, cddc_p3, cddc_p4 = (
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][0],
)
cddd_p1, cddd_p2, cddd_p3, cddd_p4 = (
    payoff_table_4pl[3][0],
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][1],
)
dccc_p1, dccc_p2, dccc_p3, dccc_p4 = (
    payoff_table_4pl[1][1],
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][0],
    payoff_table_4pl[1][0],
)
dccd_p1, dccd_p2, dccd_p3, dccd_p4 = (
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][1],
)
dcdc_p1, dcdc_p2, dcdc_p3, dcdc_p4 = (
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][0],
)
dcdd_p1, dcdd_p2, dcdd_p3, dcdd_p4 = (
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][0],
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][1],
)
ddcc_p1, ddcc_p2, ddcc_p3, ddcc_p4 = (
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][1],
    payoff_table_4pl[2][0],
    payoff_table_4pl[2][0],
)
ddcd_p1, ddcd_p2, ddcd_p3, ddcd_p4 = (
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][0],
    payoff_table_4pl[3][1],
)
dddc_p1, dddc_p2, dddc_p3, dddc_p4 = (
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][1],
    payoff_table_4pl[3][0],
)
dddd_p1, dddd_p2, dddd_p3, dddd_p4 = (
    payoff_table_4pl[4][1],
    payoff_table_4pl[4][1],
    payoff_table_4pl[4][1],
    payoff_table_4pl[4][1],
)
# states
cccc_obs = 0
cccd_obs = 1
ccdc_obs = 2
ccdd_obs = 3
cdcc_obs = 4
cdcd_obs = 5
cddc_obs = 6
cddd_obs = 7
dccc_obs = 8
dccd_obs = 9
dcdc_obs = 10
dcdd_obs = 11
ddcc_obs = 12
ddcd_obs = 13
dddc_obs = 14
dddd_obs = 15


# ###### Begin actual tests #######
@pytest.mark.parametrize("payoff", [payoff_table_2pl])
def test_single_batch_2pl(payoff) -> None:
    num_envs = 5
    rng = jax.random.PRNGKey(0)
    num_players = 2
    len_one_hot = 2**num_players + 1
    # setup
    env = IteratedTensorGameNPlayer(
        num_players=2, num_inner_steps=5, num_outer_steps=1
    )
    env_params = EnvParams(payoff_table=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)
    obs_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # test 2 player
    # cc
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action), env_params
    )
    expected_reward1 = cc_p1 * r_array
    expected_reward2 = cc_p2 * r_array
    expected_obs1 = jax.nn.one_hot(
        cc_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    expected_obs2 = jax.nn.one_hot(
        cc_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)

    # dc
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action), env_params
    )

    expected_obs1 = jax.nn.one_hot(
        dc_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    expected_obs2 = jax.nn.one_hot(
        cd_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    assert jnp.array_equal(rewards[0], dc_p1 * r_array)
    assert jnp.array_equal(rewards[1], dc_p2 * r_array)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)

    # cd
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action), env_params
    )
    expected_obs1 = jax.nn.one_hot(
        cd_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    expected_obs2 = jax.nn.one_hot(
        dc_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    assert jnp.array_equal(rewards[0], cd_p1 * r_array)
    assert jnp.array_equal(rewards[1], cd_p2 * r_array)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)

    # dd
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action), env_params
    )
    expected_obs1 = jax.nn.one_hot(
        dd_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    expected_obs2 = jax.nn.one_hot(
        dd_obs * obs_array, len_one_hot, dtype=jnp.int8
    )
    assert jnp.array_equal(rewards[0], dd_p1 * r_array)
    assert jnp.array_equal(rewards[1], dd_p2 * r_array)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)


@pytest.mark.parametrize("payoff", [payoff_table_4pl])
def test_single_batch_4pl(payoff) -> None:
    num_envs = 5
    rng = jax.random.PRNGKey(0)
    num_players = 4
    len_one_hot = 2**num_players + 1
    # setup
    env = IteratedTensorGameNPlayer(
        num_players=num_players, num_inner_steps=100, num_outer_steps=1
    )
    env_params = EnvParams(payoff_table=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)
    obs_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # test 4 player
    # cccc
    cccc = (0 * action, 0 * action, 0 * action, 0 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, cccc, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        cccc_p1 * r_array,
        cccc_p2 * r_array,
        cccc_p3 * r_array,
        cccc_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(cccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)

    # cccd
    cccd = (0 * action, 0 * action, 0 * action, 1 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, cccd, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        cccd_p1 * r_array,
        cccd_p2 * r_array,
        cccd_p3 * r_array,
        cccd_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(cccd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ccdc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cdcc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)
    # ccdc
    ccdc = (0 * action, 0 * action, 1 * action, 0 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, ccdc, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        ccdc_p1 * r_array,
        ccdc_p2 * r_array,
        ccdc_p3 * r_array,
        ccdc_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(ccdc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cdcc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cccd_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)
    # ccdd
    ccdd = (0 * action, 0 * action, 1 * action, 1 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, ccdd, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        ccdd_p1 * r_array,
        ccdd_p2 * r_array,
        ccdd_p3 * r_array,
        ccdd_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(ccdd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cddc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ddcc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dccd_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)
    # cdcc
    cdcc = (0 * action, 1 * action, 0 * action, 0 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, cdcc, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        cdcc_p1 * r_array,
        cdcc_p2 * r_array,
        cdcc_p3 * r_array,
        cdcc_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(cdcc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cccd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ccdc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)
    # cdcd
    cdcd = (0 * action, 1 * action, 0 * action, 1 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, cdcd, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        cdcd_p1 * r_array,
        cdcd_p2 * r_array,
        cdcd_p3 * r_array,
        cdcd_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(cdcd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dcdc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cdcd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dcdc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)
    # cddc skipped
    # cddd skipped
    # dccd skipped
    # dcdc skipped
    # dcdd
    dcdd = (1 * action, 0 * action, 1 * action, 1 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, dcdd, env_params
    )
    expected_reward1, expected_reward2, expected_reward3, expected_reward4 = (
        dcdd_p1 * r_array,
        dcdd_p2 * r_array,
        dcdd_p3 * r_array,
        dcdd_p4 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3, expected_obs4 = (
        jax.nn.one_hot(dcdd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cddd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(dddc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ddcd_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(rewards[3], expected_reward4)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)
    assert jnp.array_equal(obs[3], expected_obs4)
    # ddcc skipped
    # ddcd skipped
    # dddc skipped
    # dddd skipped


@pytest.mark.parametrize("payoff", [payoff_table_3pl])
def test_single_batch_3pl(payoff) -> None:
    num_envs = 5
    rng = jax.random.PRNGKey(0)
    num_players = 3
    len_one_hot = 2**num_players + 1
    # setup
    env = IteratedTensorGameNPlayer(
        num_players=num_players, num_inner_steps=100, num_outer_steps=1
    )
    env_params = EnvParams(payoff_table=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)
    obs_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # test 3 player
    # ccc
    ccc = (0 * action, 0 * action, 0 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, ccc, env_params
    )
    expected_reward1, expected_reward2, expected_reward3 = (
        ccc_p1 * r_array,
        ccc_p2 * r_array,
        ccc_p3 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3 = (
        jax.nn.one_hot(ccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ccc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)

    # dcd
    dcd = (1 * action, 0 * action, 1 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, dcd, env_params
    )
    expected_reward1, expected_reward2, expected_reward3 = (
        dcd_p1 * r_array,
        dcd_p2 * r_array,
        dcd_p3 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3 = (
        jax.nn.one_hot(dcd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cdd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ddc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)

    # dcc
    dcc = (1 * action, 0 * action, 0 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, dcc, env_params
    )
    expected_reward1, expected_reward2, expected_reward3 = (
        dcc_p1 * r_array,
        dcc_p2 * r_array,
        dcc_p3 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3 = (
        jax.nn.one_hot(dcc_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ccd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(cdc_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)

    # ddd
    ddd = (1 * action, 1 * action, 1 * action)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, ddd, env_params
    )
    expected_reward1, expected_reward2, expected_reward3 = (
        ddd_p1 * r_array,
        ddd_p2 * r_array,
        ddd_p3 * r_array,
    )
    expected_obs1, expected_obs2, expected_obs3 = (
        jax.nn.one_hot(ddd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ddd_obs * obs_array, len_one_hot, dtype=jnp.int8),
        jax.nn.one_hot(ddd_obs * obs_array, len_one_hot, dtype=jnp.int8),
    )
    assert jnp.array_equal(rewards[0], expected_reward1)
    assert jnp.array_equal(rewards[1], expected_reward2)
    assert jnp.array_equal(rewards[2], expected_reward3)
    assert jnp.array_equal(obs[0], expected_obs1)
    assert jnp.array_equal(obs[1], expected_obs2)
    assert jnp.array_equal(obs[2], expected_obs3)


def test_batch_diff_actions() -> None:
    rng = jax.random.PRNGKey(0)

    # 5 envs, with actions ccc, dcc, cdc, ddd, ccd
    pl1_actions = jnp.array([0, 1, 0, 1, 0], dtype=jnp.float32)
    pl2_actions = jnp.array([0, 0, 1, 1, 0], dtype=jnp.float32)
    pl3_actions = jnp.array([0, 0, 0, 1, 1], dtype=jnp.float32)
    given_actions = (pl1_actions, pl2_actions, pl3_actions)
    pl1_reward = jnp.array(
        [ccc_p1, dcc_p1, cdc_p1, ddd_p1, ccd_p1], dtype=jnp.float32
    )
    pl2_reward = jnp.array(
        [ccc_p2, dcc_p2, cdc_p2, ddd_p2, ccd_p2], dtype=jnp.float32
    )
    pl3_reward = jnp.array(
        [ccc_p3, dcc_p3, cdc_p3, ddd_p3, ccd_p3], dtype=jnp.float32
    )
    exp_rewards = (pl1_reward, pl2_reward, pl3_reward)
    pl1_states = jnp.array(
        [ccc_obs, dcc_obs, cdc_obs, ddd_obs, ccd_obs], dtype=jnp.float32
    )
    pl2_states = jnp.array(
        [ccc_obs, ccd_obs, dcc_obs, ddd_obs, cdc_obs], dtype=jnp.float32
    )
    pl3_states = jnp.array(
        [ccc_obs, cdc_obs, ccd_obs, ddd_obs, dcc_obs], dtype=jnp.float32
    )
    exp_obs = [
        jax.nn.one_hot(pl1_states, 9, dtype=jnp.int8),
        jax.nn.one_hot(pl2_states, 9, dtype=jnp.int8),
        jax.nn.one_hot(pl3_states, 9, dtype=jnp.int8),
    ]

    env = IteratedTensorGameNPlayer(
        num_players=3, num_inner_steps=10, num_outer_steps=1
    )
    env_params = EnvParams(payoff_table=payoff_table_3pl)
    # we want to batch over envs purely by actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)
    obs, env_state, rewards, done, info = env.step(
        rng,
        env_state,
        given_actions,
        env_params,
    )
    assert jnp.array_equal(rewards[0], exp_rewards[0])
    assert jnp.array_equal(rewards[1], exp_rewards[1])
    assert jnp.array_equal(rewards[2], exp_rewards[2])
    assert jnp.array_equal(obs[0], exp_obs[0])
    assert jnp.array_equal(obs[1], exp_obs[1])
    assert jnp.array_equal(obs[2], exp_obs[2])
    assert (done == False).all()


def test_tit_for_tat_strict_match_3pl() -> None:
    # just tests they all cooperate
    num_envs = 5
    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )
    env = IteratedTensorGameNPlayer(
        num_players=3, num_inner_steps=5, num_outer_steps=1
    )
    env_params = EnvParams(payoff_table=payoff_table_3pl)

    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    obs, env_state = env.reset(rngs, env_params)
    tit_for_tat = TitForTatStrictStay(num_envs)

    action_0, _, _ = tit_for_tat._policy(None, obs[0], None)
    action_1, _, _ = tit_for_tat._policy(None, obs[1], None)
    action_2, _, _ = tit_for_tat._policy(None, obs[2], None)
    assert jnp.array_equal(action_0, action_1)
    assert jnp.array_equal(action_0, action_2)

    for _ in range(10):
        obs, env_state, rewards, done, info = env.step(
            rngs, env_state, (action_0, action_1, action_2), env_params
        )
        assert jnp.array_equal(
            rewards[0],
            rewards[1],
        )
        assert jnp.array_equal(
            rewards[0],
            rewards[2],
        )


def test_tit_for_tat_strict_match_4pl() -> None:
    # just tests they all cooperate
    num_envs = 5
    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )
    env = IteratedTensorGameNPlayer(
        num_players=4, num_inner_steps=5, num_outer_steps=1
    )
    env_params = EnvParams(payoff_table=payoff_table_4pl)

    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    obs, env_state = env.reset(rngs, env_params)
    tit_for_tat = TitForTatStrictStay(num_envs)

    action_0, _, _ = tit_for_tat._policy(None, obs[0], None)
    action_1, _, _ = tit_for_tat._policy(None, obs[1], None)
    action_2, _, _ = tit_for_tat._policy(None, obs[2], None)
    action_3, _, _ = tit_for_tat._policy(None, obs[3], None)
    assert jnp.array_equal(action_0, action_1)
    assert jnp.array_equal(action_0, action_2)
    assert jnp.array_equal(action_0, action_3)

    for _ in range(10):
        obs, env_state, rewards, done, info = env.step(
            rngs,
            env_state,
            (action_0, action_1, action_2, action_3),
            env_params,
        )
        assert jnp.array_equal(
            rewards[0],
            rewards[1],
        )
        assert jnp.array_equal(
            rewards[0],
            rewards[2],
        )
        assert jnp.array_equal(
            rewards[0],
            rewards[3],
        )


def test_longer_game_4pl() -> None:
    num_envs = 2
    num_outer_steps = 25
    num_inner_steps = 2
    env = IteratedTensorGameNPlayer(
        num_players=4,
        num_inner_steps=num_inner_steps,
        num_outer_steps=num_outer_steps,
    )

    # batch over actions and env_states
    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    env_params = EnvParams(payoff_table=payoff_table_4pl)
    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )
    obs, env_state = env.reset(rngs, env_params)
    agent = TitForTatStrictStay(num_envs)
    r1 = []
    r2 = []
    r3 = []
    r4 = []
    for _ in range(num_outer_steps):
        for _ in range(num_inner_steps):
            action, _, _ = agent._policy(None, obs[0], None)
            obs, env_state, rewards, done, info = env.step(
                rngs, env_state, (action, action, action, action), env_params
            )
            r1.append(rewards[0])
            r2.append(rewards[1])
            r3.append(rewards[2])
            r4.append(rewards[3])
            # all coop w/ this strategy
            assert jnp.array_equal(rewards[0], rewards[1])
            assert jnp.array_equal(rewards[0], rewards[2])
            assert jnp.array_equal(rewards[0], rewards[3])

    assert (done == True).all()

    assert jnp.mean(jnp.stack(r1)) == 4
    assert jnp.mean(jnp.stack(r2)) == 4
    assert jnp.mean(jnp.stack(r3)) == 4


def test_done():
    num_inner_steps = 5
    env = IteratedTensorGameNPlayer(
        num_players=4, num_inner_steps=num_inner_steps, num_outer_steps=1
    )
    env_params = EnvParams(
        payoff_table=payoff_table_4pl,
    )
    rng = jax.random.PRNGKey(0)
    obs, env_state = env.reset(rng, env_params)
    action = 0

    for _ in range(num_inner_steps - 1):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action, action, action), env_params
        )
        # check not start state
        assert (done == False).all()
        assert (obs[0].argmax() != 8).all()
        assert (obs[1].argmax() != 8).all()
        assert (obs[2].argmax() != 8).all()
        assert (obs[3].argmax() != 8).all()

    # check final
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (action, action, action, action), env_params
    )
    assert (done == True).all()

    # check back at start
    assert jnp.array_equal(obs[0].argmax(), 2**4)
    assert jnp.array_equal(obs[1].argmax(), 2**4)
    assert jnp.array_equal(obs[2].argmax(), 2**4)
    assert jnp.array_equal(obs[3].argmax(), 2**4)


def test_reset():
    rng = jax.random.PRNGKey(0)
    env = IteratedTensorGameNPlayer(
        num_players=4, num_inner_steps=5, num_outer_steps=20
    )
    env_params = EnvParams(
        payoff_table=payoff_table_4pl,
    )
    action = 0

    obs, env_state = env.reset(rng, env_params)
    # one fewer than inner steps
    for _ in range(4):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action, action, action), env_params
        )
        assert done == False

    obs, env_state = env.reset(rng, env_params)
    # assert not done bc we reset env
    for _ in range(4):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action, action, action), env_params
        )
        assert done == False
