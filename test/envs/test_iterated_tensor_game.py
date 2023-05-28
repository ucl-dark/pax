import jax
import jax.numpy as jnp
import pytest

from pax.agents.strategies import TitForTat
from pax.agents.tensor_strategies import TitForTatStrictStay
from pax.envs.iterated_tensor_game import EnvParams, IteratedTensorGame

# payoff matrix accroding to
# https://www.classes.cs.uchicago.edu/archive/1998/fall/CS105/Project/node6.html
payoff = [
    [7, 7, 7],
    [3, 3, 9],
    [3, 9, 3],
    [0, 5, 5],
    [9, 3, 3],
    [5, 0, 5],
    [5, 5, 0],
    [1, 1, 1],
]

ccc_p1 = payoff[0][0]
ccc_p2 = payoff[0][1]
ccc_p3 = payoff[0][2]
ccd_p1 = payoff[1][0]
ccd_p2 = payoff[1][1]
ccd_p3 = payoff[1][2]
cdc_p1 = payoff[2][0]
cdc_p2 = payoff[2][1]
cdc_p3 = payoff[2][2]
cdd_p1 = payoff[3][0]
cdd_p2 = payoff[3][1]
cdd_p3 = payoff[3][2]
dcc_p1 = payoff[4][0]
dcc_p2 = payoff[4][1]
dcc_p3 = payoff[4][2]
dcd_p1 = payoff[5][0]
dcd_p2 = payoff[5][1]
dcd_p3 = payoff[5][2]
ddc_p1 = payoff[6][0]
ddc_p2 = payoff[6][1]
ddc_p3 = payoff[6][2]
ddd_p1 = payoff[7][0]
ddd_p2 = payoff[7][1]
ddd_p3 = payoff[7][2]


@pytest.mark.parametrize("payoff", [payoff])
def test_single_batch_rewards(payoff) -> None:
    num_envs = 5
    rng = jax.random.PRNGKey(0)
    env = IteratedTensorGame(num_inner_steps=5, num_outer_steps=1)
    env_params = EnvParams(payoff_matrix=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # first step
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ccc_p1 * r_array)
    assert jnp.array_equal(rewards[1], ccc_p2 * r_array)
    assert jnp.array_equal(rewards[2], ccc_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ccd_p1 * r_array)
    assert jnp.array_equal(rewards[1], ccd_p2 * r_array)
    assert jnp.array_equal(rewards[2], ccd_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cdc_p1 * r_array)
    assert jnp.array_equal(rewards[1], cdc_p2 * r_array)
    assert jnp.array_equal(rewards[2], cdc_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cdd_p1 * r_array)
    assert jnp.array_equal(rewards[1], cdd_p2 * r_array)
    assert jnp.array_equal(rewards[1], cdd_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dcc_p1 * r_array)
    assert jnp.array_equal(rewards[1], dcc_p2 * r_array)
    assert jnp.array_equal(rewards[2], dcc_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dcd_p1 * r_array)
    assert jnp.array_equal(rewards[1], dcd_p2 * r_array)
    assert jnp.array_equal(rewards[2], dcd_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ddc_p1 * r_array)
    assert jnp.array_equal(rewards[1], ddc_p2 * r_array)
    assert jnp.array_equal(rewards[2], ddc_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ddd_p1 * r_array)
    assert jnp.array_equal(rewards[1], ddd_p2 * r_array)
    assert jnp.array_equal(rewards[1], ddd_p3 * r_array)


testdata = [
    ((0, 0, 0), (7, 7, 7), [payoff]),
    ((0, 0, 1), (3, 3, 9), [payoff]),
    ((0, 1, 0), (3, 9, 3), [payoff]),
    ((0, 1, 1), (0, 5, 5), [payoff]),
    ((1, 0, 0), (9, 3, 3), [payoff]),
    ((1, 0, 1), (5, 0, 5), [payoff]),
    ((1, 1, 0), (5, 5, 0), [payoff]),
    ((1, 1, 1), (1, 1, 1), [payoff]),
]


@pytest.mark.parametrize("actions, expected_rewards, payoff_list", testdata)
def test_batch_outcomes(actions, expected_rewards, payoff_list) -> None:
    num_envs = 3
    all_ones = jnp.ones((num_envs,))

    rng = jax.random.PRNGKey(0)

    a1, a2, a3 = actions
    expected_r1, expected_r2, expected_r3 = expected_rewards

    env = IteratedTensorGame(num_inner_steps=5, num_outer_steps=1)
    env_params = EnvParams(payoff_matrix=payoff)
    # we want to batch over envs purely by actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)
    obs, env_state, rewards, done, info = env.step(
        rng,
        env_state,
        (a1 * all_ones, a2 * all_ones, a3 * all_ones),
        env_params,
    )

    assert jnp.array_equal(rewards[0], expected_r1 * jnp.ones((num_envs,)))
    assert jnp.array_equal(rewards[1], expected_r2 * jnp.ones((num_envs,)))
    assert jnp.array_equal(rewards[2], expected_r3 * jnp.ones((num_envs,)))
    assert (done == False).all()


def test_batch_by_rngs() -> None:
    # we don't use the rng in step but good test of how runners wil use env
    num_envs = 2
    rng = jnp.concatenate(
        [jax.random.PRNGKey(0), jax.random.PRNGKey(0)]
    ).reshape(num_envs, -1)
    env = IteratedTensorGame(num_inner_steps=5, num_outer_steps=1)
    env_params = EnvParams(payoff_matrix=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over envs purely by actions
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # first step
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ccc_p1 * r_array)
    assert jnp.array_equal(rewards[1], ccc_p2 * r_array)
    assert jnp.array_equal(rewards[2], ccc_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ccd_p1 * r_array)
    assert jnp.array_equal(rewards[1], ccd_p2 * r_array)
    assert jnp.array_equal(rewards[2], ccd_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cdc_p1 * r_array)
    assert jnp.array_equal(rewards[1], cdc_p2 * r_array)
    assert jnp.array_equal(rewards[2], cdc_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cdd_p1 * r_array)
    assert jnp.array_equal(rewards[1], cdd_p2 * r_array)
    assert jnp.array_equal(rewards[1], cdd_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dcc_p1 * r_array)
    assert jnp.array_equal(rewards[1], dcc_p2 * r_array)
    assert jnp.array_equal(rewards[2], dcc_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dcd_p1 * r_array)
    assert jnp.array_equal(rewards[1], dcd_p2 * r_array)
    assert jnp.array_equal(rewards[2], dcd_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ddc_p1 * r_array)
    assert jnp.array_equal(rewards[1], ddc_p2 * r_array)
    assert jnp.array_equal(rewards[2], ddc_p3 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], ddd_p1 * r_array)
    assert jnp.array_equal(rewards[1], ddd_p2 * r_array)
    assert jnp.array_equal(rewards[1], ddd_p3 * r_array)


def test_tit_for_tat_strict_match() -> None:
    # just tests they all cooperate
    num_envs = 5
    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )
    env = IteratedTensorGame(num_inner_steps=5, num_outer_steps=1)
    env_params = EnvParams(payoff_matrix=payoff)

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


def test_longer_game() -> None:
    num_envs = 1
    num_outer_steps = 25
    num_inner_steps = 2
    env = IteratedTensorGame(
        num_inner_steps=num_inner_steps, num_outer_steps=num_outer_steps
    )

    # batch over actions and env_states
    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    env_params = EnvParams(
        payoff_matrix=payoff,
    )

    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )

    obs, env_state = env.reset(rngs, env_params)

    agent = TitForTatStrictStay(num_envs)
    r1 = []
    r2 = []
    r3 = []
    for _ in range(num_outer_steps):
        for _ in range(num_inner_steps):
            action, _, _ = agent._policy(None, obs[0], None)
            obs, env_state, rewards, done, info = env.step(
                rngs, env_state, (action, action, action), env_params
            )
            r1.append(rewards[0])
            r2.append(rewards[1])
            r3.append(rewards[2])
            assert jnp.array_equal(rewards[0], rewards[1])
            assert jnp.array_equal(rewards[0], rewards[2])
    assert (done == True).all()

    assert jnp.mean(jnp.stack(r1)) == 7
    assert jnp.mean(jnp.stack(r2)) == 7
    assert jnp.mean(jnp.stack(r3)) == 7


def test_done():
    num_inner_steps = 5
    env = IteratedTensorGame(
        num_inner_steps=num_inner_steps, num_outer_steps=1
    )
    env_params = EnvParams(
        payoff_matrix=payoff,
    )
    rng = jax.random.PRNGKey(0)
    obs, env_state = env.reset(rng, env_params)
    action = 0

    for _ in range(num_inner_steps - 1):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action, action), env_params
        )
        assert (done == False).all()
        assert (obs[0].argmax() != 8).all()
        assert (obs[1].argmax() != 8).all()
        assert (obs[2].argmax() != 8).all()

    # check final
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (action, action, action), env_params
    )
    assert (done == True).all()

    # check back at start
    assert jnp.array_equal(obs[0].argmax(), 8)
    assert jnp.array_equal(obs[1].argmax(), 8)
    assert jnp.array_equal(obs[2].argmax(), 8)


def test_reset():
    rng = jax.random.PRNGKey(0)
    env = IteratedTensorGame(num_inner_steps=5, num_outer_steps=20)
    env_params = EnvParams(
        payoff_matrix=payoff,
    )
    action = 0

    obs, env_state = env.reset(rng, env_params)
    for _ in range(4):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action, action), env_params
        )
        assert done == False

    obs, env_state = env.reset(rng, env_params)

    for _ in range(4):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action, action), env_params
        )
        assert done == False
