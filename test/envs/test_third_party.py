import jax
import jax.numpy as jnp
import pytest

from pax.envs.third_party_punishment import ThirdPartyPunishment, EnvParams

# PAYOFF AND PUNISHMENT SETUP
payoff= [
    [ -1 , 1000 ],
    [-3 , 0 ],
    [1000 , -2],
]
punishment= -3

cc_p1 = payoff[0][0]
cc_p2 = payoff[0][0]
cd_p1 = payoff[1][0]
cd_p2 = payoff[1][1]
dc_p1 = payoff[1][1]
dc_p2 = payoff[1][0]
dd_p1 = payoff[2][1]
dd_p2 = payoff[2][1]



# actions cheatsheet
    # CC_no_P = 0
    # CC_P1 = 1
    # CC_P2 = 2
    # CC_both_P = 3

    # CD_no_P = 4
    # CD_P1 = 5
    # CD_P2 = 6
    # CD_both_P = 7

    # DC_no_P = 8
    # DC_P1 = 9
    # DC_P2 = 10
    # DC_both_P = 11

    # DD_no_P = 12
    # DD_P1 = 13
    # DD_P2 = 14
    # DD_both_P = 15

# games pairing cheatsheet:
# pl1 (1st game) vs pl2 (1st game), pl3 punishes
# pl2 (2nd game) vs pl3 (1st games), pl1 punishes
# pl3 (2nd game) vs pl1 (2nd game), pl2 punishes

# fpp observed states cheatsheet:
# pl1:  pl1 vs pl2, pl1 vs pl3, pl2 vs pl3
# pl2:  pl2 vs pl3, pl2 vs pl1, pl3 vs pl1
# pl3:  pl3 vs pl1, pl3 vs pl2, pl1 vs pl2


################################################
# we test that:
# the correct ipd rewards are given based on prev step
# punishments are applied correctly to the reward based on current step
# states are returned correctly based on current step

# this is done by cross-testing 6 cases of prev step ipd actions and 4 cases of curr step punishment actions
################################################

@pytest.mark.parametrize("payoff", [payoff])
def test_single_batch_rewards(payoff) -> None:
    num_envs = 5
    rng = jax.random.PRNGKey(0)
    env = ThirdPartyPunishment(num_inner_steps=10, num_outer_steps=1)
    env_params = EnvParams(payoff_table=payoff, punishment=punishment)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)


    # we want to batch over actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    ####################### Cross-Testing pairing ipd and punishment to 3 games ############################
    # IPD ACTIONS CASE 1
    # all games cc 
    prev_actions1 = (0 * action, 1 * action, 3 * action)
    pl1_rew1 = 2*cc_p1 * r_array
    pl2_rew1 = 2*cc_p1 * r_array
    pl3_rew1 = 2*cc_p1 * r_array
    obs1 = (0* r_array,0* r_array,0* r_array)
    obs1 = tuple(jax.nn.one_hot(obs1, 2**6+1, dtype=jnp.int8))
    test1 = (prev_actions1, pl1_rew1, pl2_rew1, pl3_rew1)

    # IPD ACTIONS CASE 2
    # all games dd
    prev_actions2 = (12 * action, 14 * action, 15 * action)
    pl1_rew2 = 2*dd_p1 * r_array
    pl2_rew2 = 2*dd_p1 * r_array
    pl3_rew2 = 2*dd_p1 * r_array
    obs2 = (63* r_array,63* r_array,63* r_array)
    obs2 = tuple(jax.nn.one_hot(obs2, 2**6+1, dtype=jnp.int8))
    test2 = (prev_actions2, pl1_rew2, pl2_rew2, pl3_rew2)

    # IPD ACTIONS CASE 3
    # cc, cd, dd actions for each 3 players
    # 3 games are cc, dd, dc 
    prev_actions3 = (2 * action, 5 * action, 15 * action)
    pl1_rew3 = cc_p1 * r_array + dc_p2*r_array
    pl2_rew3 = cc_p2 * r_array + dd_p1*r_array
    pl3_rew3 = dd_p2 * r_array + dc_p1*r_array
    # cccddd,  ddccdc, dcddcc 
    obs3 = ((2+4+1)* r_array, (2+16+32)* r_array, (4+8+32)* r_array)
    obs3 = tuple(jax.nn.one_hot(obs3, 2**6+1, dtype=jnp.int8))
    test3 = (prev_actions3, pl1_rew3, pl2_rew3, pl3_rew3)

    # IPD ACTIONS CASE 4
    # cd, dc, dc actions for each 3 players
    # 3 games are cd, cd, cd
    prev_actions4 = (6 * action, 9 * action, 11 * action)
    pl1_rew4 = cd_p1 * r_array + cd_p2*r_array
    pl2_rew4 = cd_p1 * r_array + cd_p2*r_array
    pl3_rew4 = cd_p1 * r_array + cd_p2*r_array
    # cddccd, cddccd, cddccd
    obs4 = ((1+8+16)* r_array, (1+8+16)* r_array, (1+8+16)* r_array)
    obs4 = tuple(jax.nn.one_hot(obs4, 2**6+1, dtype=jnp.int8))
    test4 = (prev_actions4, pl1_rew4, pl2_rew4, pl3_rew4)

    # IPD ACTIONS CASE 5
    # dd, dc, cd actions for each 3 players
    # 3 games are dd, cc, dd
    prev_actions5 = (13 * action, 8 * action, 5 * action)
    pl1_rew5 = dd_p1 * r_array + dd_p2*r_array
    pl2_rew5 = cc_p1 * r_array + dd_p2*r_array
    pl3_rew5 = dd_p1 * r_array + cc_p2*r_array
    # ddddcc, ccdddd, ddccdd
    obs5 = ((4+8+16+32)* r_array, (1+4+8+2)* r_array, (1+2+32+16)* r_array)
    obs5 = tuple(jax.nn.one_hot(obs5, 2**6+1, dtype=jnp.int8))
    test5 = (prev_actions5, pl1_rew5, pl2_rew5, pl3_rew5)

    # IPD ACTIONS CASE 6
    # dc, cc, cd actions for each 3 players
    # 3 games are dc, cc, dc
    prev_actions6 = (10 * action, 3 * action, 7 * action)
    pl1_rew6 = dc_p1 * r_array + dc_p2*r_array
    pl2_rew6 = cc_p1 * r_array + dc_p2*r_array
    pl3_rew6 = dc_p1 * r_array + cc_p2*r_array
    # dccdcc, cccddc, dcccdc
    obs6 = ((4+32)* r_array, (2+4)* r_array, (2+32)* r_array)
    obs6 = tuple(jax.nn.one_hot(obs6, 2**6+1, dtype=jnp.int8))
    test6 = (prev_actions6, pl1_rew6, pl2_rew6, pl3_rew6)

   ############### CHECKING REWARDS #################

    # # PUNISHMENT CASE 1
    # # no p, no p, no p-> pl1 0x, pl2 0x, pl3 0x
    punish_action_case1_1 = (0 * action, 0 * action, 0 * action)
    punish_action_case1_2 = (4 * action, 0 * action, 8 * action)
    punish_action_case1_3 = (8 * action, 4 * action, 12 * action)
    punish_action_case1_4 = (12 * action, 0 * action, 4 * action)
    punish_action_case1 = [punish_action_case1_1,punish_action_case1_2,punish_action_case1_3,punish_action_case1_4]

    for (prev_action, rew1, rew2, rew3) in [test1, test2]:#, test3, test4, test5, test6]:
        for punish_action in punish_action_case1:
            (obs,_), env_state, rewards, done, info = env.step(
                rng, env_state, (prev_action, punish_action), env_params
            )
            assert jnp.array_equal(rewards[0], rew1 + 0*punishment*r_array) 
            assert jnp.array_equal(rewards[1], rew2 + 0*punishment*r_array )
            assert jnp.array_equal(rewards[2], rew3 + 0*punishment*r_array )

    # PUNISHMENT CASE 2
    # no p, both p, p1 -> pl1 2x, pl2 0x, pl3 1x
    punish_action_case2_1 = (0 * action, 3 * action, 1 * action)
    punish_action_case2_2 = (4 * action, 7 * action, 9 * action)
    punish_action_case2_3 = (8 * action, 11 * action, 13 * action)
    punish_action_case2_4 = (12 * action, 15 * action, 5 * action)
    punish_action_case2 = [punish_action_case2_1,punish_action_case2_2,punish_action_case2_3,punish_action_case2_4]

    for (prev_action, rew1, rew2, rew3) in [test1, test2, test3, test4, test5, test6]:
        for punish_action in punish_action_case2:
            (obs,_), env_state, rewards, done, info = env.step(
                rng, env_state, (prev_action, punish_action), env_params
            )

            assert jnp.array_equal(rewards[0], rew1 + 2*punishment*r_array) 
            assert jnp.array_equal(rewards[1], rew2 + 0*punishment*r_array )
            assert jnp.array_equal(rewards[2], rew3 + 1*punishment*r_array )

    # PUNISHMENT CASE 3
    # pl2, pl3, pl1
    punish_action_case3_1 = (1 * action, 1 * action, 1 * action)
    punish_action_case3_2 = (5 * action, 9 * action, 13 * action)
    punish_action_case3_3 = (9 * action, 1 * action, 13 * action)
    punish_action_case3_4 = (1 * action, 5 * action, 5 * action)
    punish_action_case3 = [punish_action_case3_1,punish_action_case3_2,punish_action_case3_3,punish_action_case3_4]

    for (prev_action, rew1, rew2, rew3) in [test1, test2, test3, test4, test5, test6]:
        for punish_action in punish_action_case3:
            (obs,_), env_state, rewards, done, info = env.step(
                rng, env_state, (prev_action, punish_action), env_params
            )

            assert jnp.array_equal(rewards[0], rew1 + 1*punishment*r_array) 
            assert jnp.array_equal(rewards[1], rew2 + 1*punishment*r_array )
            assert jnp.array_equal(rewards[2], rew3 + 1*punishment*r_array )

    # PUNISHMENT CASE 4
    # pl2, both, pl2
    punish_action_case4_1 = (1 * action, 15 * action, 2 * action)
    punish_action_case4_2 = (5 * action, 11 * action, 10 * action)
    punish_action_case4_3 = (9 * action, 7 * action, 6 * action)
    punish_action_case4_4 = (13 * action, 3 * action, 14 * action)
    punish_action_case4 = [punish_action_case4_1,punish_action_case4_2,punish_action_case4_3,punish_action_case4_4]

    for (prev_action, rew1, rew2, rew3) in [test1, test2, test3, test4, test5, test6]:
        for punish_action in punish_action_case4:
            (obs,_), env_state, rewards, done, info = env.step(
                rng, env_state, (prev_action, punish_action), env_params
            )

            assert jnp.array_equal(rewards[0], rew1 + 1*punishment*r_array) 
            assert jnp.array_equal(rewards[1], rew2 + 2*punishment*r_array )
            assert jnp.array_equal(rewards[2], rew3 + 1*punishment*r_array )

    ############### CHECKING OBSERVED STATES #################
    prev_action = (0 * action, 0 * action, 0 * action)
    for curr_action, exp_obs in zip([ prev_actions1, prev_actions2, prev_actions3, prev_actions4, prev_actions5, prev_actions6], [obs1, obs2, obs3, obs4, obs5, obs6]):
        (obs,_), env_state, rewards, done, info = env.step(
            rng, env_state, (prev_action, curr_action), env_params
        )
        assert jnp.array_equal(obs[0], exp_obs[0])
        assert jnp.array_equal(obs[1], exp_obs[1])
        assert jnp.array_equal(obs[2], exp_obs[2])
        

def test_longer_game() -> None:
    num_envs = 1
    num_outer_steps = 25
    num_inner_steps = 2
    env = ThirdPartyPunishment(num_inner_steps, num_outer_steps)
    env_params = EnvParams(payoff_table=payoff, punishment=punishment)

    # batch over actions and env_states
    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )

    obs, env_state = env.reset(rngs, env_params)

    r1 = []
    r2 = []
    r3 = []
    for _ in range(num_outer_steps):
        for _ in range(num_inner_steps):
            action_arr = jnp.ones((num_envs,), dtype=jnp.float32)
            # pl1 not punished, pl2 punished 2x, pl3 punished 2x
            action=  (3 * action_arr, 1 * action_arr, 2 * action_arr)
            (obs,_), env_state, rewards, done, info = env.step(
                rngs, env_state, (action, action), env_params
            )
            r1.append(rewards[0])
            r2.append(rewards[1])
            r3.append(rewards[2])
            assert jnp.array_equal(rewards[1], rewards[2])
    assert (done == True).all()

    assert jnp.mean(jnp.stack(r1)) == -2
    assert jnp.mean(jnp.stack(r2)) == -8
    assert jnp.mean(jnp.stack(r3)) == -8


def test_done():
    num_inner_steps = 5
    env = ThirdPartyPunishment(num_inner_steps, 1)
    env_params = EnvParams(payoff_table=payoff, punishment=punishment)
    rng = jax.random.PRNGKey(0)
    obs, env_state = env.reset(rng, env_params)
    action_arr = jnp.ones((1,), dtype=jnp.float32)
    action=  (3 * action_arr, 1 * action_arr, 2 * action_arr)

    for _ in range(num_inner_steps - 1):
        (obs,_), env_state, rewards, done, info = env.step(
            rng, env_state, (action, action), env_params
        )
        assert (done == False).all()
        assert (obs[0].argmax() != 64).all()
        assert (obs[1].argmax() != 64).all()
        assert (obs[2].argmax() != 64).all()

    # check final
    (obs,_), env_state, rewards, done, info = env.step(
        rng, env_state, (action, action), env_params
    )
    assert (done == True).all()

    # check back at start
    assert jnp.array_equal(obs[0].argmax(), 64)
    assert jnp.array_equal(obs[1].argmax(), 64)
    assert jnp.array_equal(obs[2].argmax(), 64)


def test_reset():
    rng = jax.random.PRNGKey(0)
    env = ThirdPartyPunishment(5, 20)
    env_params = EnvParams(payoff_table=payoff, punishment=punishment)
    action_arr = jnp.ones((1,), dtype=jnp.float32)
    action=  (3 * action_arr, 1 * action_arr, 2 * action_arr)

    obs, env_state = env.reset(rng, env_params)
    for _ in range(4):
        (obs,_), env_state, rewards, done, info = env.step(
            rng, env_state, (action, action), env_params
        )
        assert done == False

    obs, env_state = env.reset(rng, env_params)

    for _ in range(4):
        (obs,_), env_state, rewards, done, info = env.step(
            rng, env_state, (action, action), env_params
        )
        assert done == False
