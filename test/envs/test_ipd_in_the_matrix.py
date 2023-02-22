import jax
import jax.numpy as jnp

from pax.envs.ipd_in_the_matrix import (
    IPDInTheMatrix,
    EnvParams,
    EnvState,
)


def test_ipditm_shapes():
    rng = jax.random.PRNGKey(0)
    env = IPDInTheMatrix(
        num_inner_steps=8, num_outer_steps=2, fixed_coin_location=True
    )

    params = EnvParams(
        payoff_matrix=jnp.array([[[1, -2], [1, -2]], [[1, -2], [1, -2]]]),
        freeze_penalty=5,
    )

    obs, state = env.reset(rng, params)
    action = jnp.ones((), dtype=int)

    assert obs[0]["observation"].shape == (5, 5, 10)
    assert obs[1]["observation"].shape == (5, 5, 10)

    assert obs[0]["inventory"].shape == (6,)
    assert obs[1]["inventory"].shape == (6,)
    obs, new_state, rewards, done, info = env.step(
        rng, state, (action, action), params
    )
    assert rewards[0].shape == ()
    assert rewards[1].shape == ()


def test_ipditm_collision():
    rng = jax.random.PRNGKey(0)
    env = IPDInTheMatrix(142, 100, True)
    params = EnvParams(
        payoff_matrix=jnp.array([[[1, -2], [1, -2]], [[1, -2], [1, -2]]]),
        freeze_penalty=5,
    )

    _, state = env.reset(rng, params)

    state = EnvState(
        red_pos=jnp.array([0, 7, 0], dtype=jnp.int8),
        blue_pos=jnp.array([1, 7, 3], dtype=jnp.int8),
        inner_t=jnp.array(42, dtype=jnp.int32),
        outer_t=jnp.array(5, dtype=jnp.int32),
        grid=jnp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 4, 3, 0, 0, 3, 4, 2],
                [0, 3, 4, 0, 0, 4, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 4, 0, 0, 4, 3, 0],
                [0, 4, 3, 0, 0, 3, 4, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=jnp.int8,
        ),
        red_inventory=jnp.array([1.0, 1.0], dtype=jnp.float32),
        blue_inventory=jnp.array([1.0, 1.0], dtype=jnp.float32),
        red_coins=jnp.array(
            [[1, 2], [2, 1], [1, 5], [2, 6], [6, 2], [5, 1]],
            dtype=jnp.int8,
        ),
        blue_coins=jnp.array(
            [[1, 1], [2, 2], [1, 6], [2, 5], [6, 1], [5, 2]],
            dtype=jnp.int8,
        ),
        freeze=jnp.array(-1, dtype=jnp.int16),
    )

    a1 = 2
    a2 = 2

    obs, new_state, _, _, _ = env.step(rng, state, (a1, a2), params)
    assert (new_state.red_pos[:2] != new_state.blue_pos[:2]).any()
