import jax
import jax.numpy as jnp
import pytest

from pax.agents.strategies import TitForTat
from pax.envs.iterated_matrix_game import EnvParams, IteratedMatrixGame

# payoff matrices for four games
ipd = [[2, 2], [0, 3], [3, 0], [1, 1]]
stag = [[4, 4], [1, 3], [3, 1], [2, 2]]
sexes = [[3, 2], [0, 0], [0, 0], [2, 3]]
chicken = [[0, 0], [-1, 1], [1, -1], [-2, -2]]
test_payoffs = [ipd, stag, sexes, chicken]


@pytest.mark.parametrize("payoff", test_payoffs)
def test_single_batch_rewards(payoff) -> None:
    num_envs = 5
    rng = jax.random.PRNGKey(0)
    env = IteratedMatrixGame(num_inner_steps=5)
    env_params = EnvParams(payoff_matrix=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # payoffs
    cc_p1, cc_p2 = payoff[0][0], payoff[0][1]
    cd_p1, cd_p2 = payoff[1][0], payoff[1][1]
    dc_p1, dc_p2 = payoff[2][0], payoff[2][1]
    dd_p1, dd_p2 = payoff[3][0], payoff[3][1]

    # first step
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cc_p1 * r_array)
    assert jnp.array_equal(rewards[1], cc_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dc_p1 * r_array)
    assert jnp.array_equal(rewards[1], dc_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cd_p1 * r_array)
    assert jnp.array_equal(rewards[1], cd_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dd_p1 * r_array)
    assert jnp.array_equal(rewards[1], dd_p2 * r_array)


testdata = [
    ((0, 0), (2, 2), ipd),
    ((1, 0), (3, 0), ipd),
    ((0, 1), (0, 3), ipd),
    ((1, 1), (1, 1), ipd),
    ((0, 0), (4, 4), stag),
    ((1, 0), (3, 1), stag),
    ((0, 1), (1, 3), stag),
    ((1, 1), (2, 2), stag),
    ((0, 0), (3, 2), sexes),
    ((1, 0), (0, 0), sexes),
    ((0, 1), (0, 0), sexes),
    ((1, 1), (2, 3), sexes),
    ((0, 0), (0, 0), chicken),
    ((1, 0), (1, -1), chicken),
    ((0, 1), (-1, 1), chicken),
    ((1, 1), (-2, -2), chicken),
]


@pytest.mark.parametrize("actions, expected_rewards, payoff", testdata)
def test_batch_outcomes(actions, expected_rewards, payoff) -> None:
    num_envs = 3
    all_ones = jnp.ones((num_envs,))

    rng = jax.random.PRNGKey(0)

    a1, a2 = actions
    expected_r1, expected_r2 = expected_rewards

    env = IteratedMatrixGame(num_inner_steps=5)
    env_params = EnvParams(payoff_matrix=payoff)
    # we want to batch over envs purely by actions
    env.step = jax.vmap(
        env.step, in_axes=(None, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (a1 * all_ones, a2 * all_ones), env_params
    )

    assert jnp.array_equal(rewards[0], expected_r1 * jnp.ones((num_envs,)))
    assert jnp.array_equal(rewards[1], expected_r2 * jnp.ones((num_envs,)))
    assert (done == False).all()


def test_batch_by_rngs() -> None:
    # we don't use the rng in step but good test of how runners wil use env
    num_envs = 2
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]

    rng = jnp.concatenate(
        [jax.random.PRNGKey(0), jax.random.PRNGKey(0)]
    ).reshape(num_envs, -1)
    env = IteratedMatrixGame(num_inner_steps=5)
    env_params = EnvParams(payoff_matrix=payoff)

    action = jnp.ones((num_envs,), dtype=jnp.float32)
    r_array = jnp.ones((num_envs,), dtype=jnp.float32)

    # we want to batch over envs purely by actions
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )
    obs, env_state = env.reset(rng, env_params)

    # payoffs
    cc_p1, cc_p2 = payoff[0][0], payoff[0][1]
    cd_p1, cd_p2 = payoff[1][0], payoff[1][1]
    dc_p1, dc_p2 = payoff[2][0], payoff[2][1]
    dd_p1, dd_p2 = payoff[3][0], payoff[3][1]

    # first step
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cc_p1 * r_array)
    assert jnp.array_equal(rewards[1], cc_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 0 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dc_p1 * r_array)
    assert jnp.array_equal(rewards[1], dc_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (0 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], cd_p1 * r_array)
    assert jnp.array_equal(rewards[1], cd_p2 * r_array)

    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (1 * action, 1 * action), env_params
    )
    assert jnp.array_equal(rewards[0], dd_p1 * r_array)
    assert jnp.array_equal(rewards[1], dd_p2 * r_array)


def test_tit_for_tat_match() -> None:
    num_envs = 5
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    rngs = jnp.concatenate(num_envs * [jax.random.PRNGKey(0)]).reshape(
        num_envs, -1
    )
    env = IteratedMatrixGame(num_inner_steps=5)
    env_params = EnvParams(payoff_matrix=payoff)

    env.reset = jax.vmap(env.reset, in_axes=(0, None), out_axes=(0, None))
    env.step = jax.vmap(
        env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
    )

    obs, env_state = env.reset(rngs, env_params)
    tit_for_tat = TitForTat(num_envs)

    action_0, _, _ = tit_for_tat._policy(None, obs[0], None)
    action_1, _, _ = tit_for_tat._policy(None, obs[1], None)
    assert jnp.array_equal(action_0, action_1)

    for _ in range(10):
        obs, env_state, rewards, done, info = env.step(
            rngs, env_state, (action_0, action_1), env_params
        )
        assert jnp.array_equal(rewards[0], rewards[1])


def test_longer_game() -> None:
    num_envs = 1
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    num_outer_steps = 25
    num_inner_steps = 2
    env = IteratedMatrixGame(num_inner_steps=num_inner_steps)

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

    agent = TitForTat(num_envs)
    r1 = []
    r2 = []
    for _ in range(num_outer_steps):
        for _ in range(num_inner_steps):
            action, _, _ = agent._policy(None, obs[0], None)
            obs, env_state, rewards, done, info = env.step(
                rngs, env_state, (action, action), env_params
            )
            r1.append(rewards[0])
            r2.append(rewards[1])
            assert jnp.array_equal(rewards[0], rewards[1])
        assert (done == True).all()

    assert jnp.mean(jnp.stack(r1)) == 2
    assert jnp.mean(jnp.stack(r2)) == 2


def test_done():
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    num_inner_steps = 5
    env = IteratedMatrixGame(num_inner_steps=num_inner_steps)
    env_params = EnvParams(
        payoff_matrix=payoff,
    )
    rng = jax.random.PRNGKey(0)
    obs, env_state = env.reset(rng, env_params)
    action = 0

    for _ in range(num_inner_steps - 1):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action), env_params
        )
        assert (done == False).all()
        assert (obs[0].argmax() != 4).all()
        assert (obs[1].argmax() != 4).all()

    # check final
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (action, action), env_params
    )
    assert (done == True).all()

    # check back at start
    assert jnp.array_equal(obs[0].argmax(), 4)
    assert jnp.array_equal(obs[1].argmax(), 4)


def test_reset():
    payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
    rng = jax.random.PRNGKey(0)
    env = IteratedMatrixGame(num_inner_steps=5)
    env_params = EnvParams(
        payoff_matrix=payoff,
    )
    action = 0

    obs, env_state = env.reset(rng, env_params)
    for _ in range(4):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action), env_params
        )
        assert done == False
        assert done == False

    obs, env_state = env.reset(rng, env_params)

    for _ in range(4):
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (action, action), env_params
        )
        assert done == False
        assert done == False
