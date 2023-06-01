import jax
import jax.numpy as jnp

from pax.envs.in_the_matrix_nplayer2 import (
    InTheMatrix,
    EnvParams,
    EnvState,
)


def test_ipditm_shapes():
    rng = jax.random.PRNGKey(0)
    env = InTheMatrix(
        num_inner_steps=8, num_outer_steps=2, fixed_coin_location=True, num_agents=2
    )

    params = EnvParams(
        payoff_matrix=jnp.array([[[1, -2], [1, -2]], [[1, -2], [1, -2]]]),
        freeze_penalty=5,
    )

    obs, state = env.reset(rng, params)
    action = jnp.ones((), dtype=int)

    assert obs["observations"][0].shape == (5, 5, 16)
    assert obs["observations"][1].shape == (5, 5, 16)

    assert obs["inventory"].shape == (5,2)
    assert obs["inventory"].shape == (5,2)
    obs, new_state, rewards, done, info = env.step(
        rng, state, (action, action), params
    )
    assert rewards[0].shape == ()
    assert rewards[1].shape == ()


def test_ipditm_collision():
    rng = jax.random.PRNGKey(0)
    env = InTheMatrix(num_inner_steps=142, num_outer_steps=100, fixed_coin_location=True, num_agents=2)
    params = EnvParams(
        payoff_matrix=jnp.array([[[1, -2], [1, -2]], [[1, -2], [1, -2]]]),
        freeze_penalty=5,
    )

    _, state = env.reset(rng, params)

    state = EnvState(
        agent_positions=[jnp.array([0, 7, 0], dtype=jnp.int8), jnp.array([1, 7, 3], dtype=jnp.int8)],
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
        agent_inventories=[jnp.array([1.0, 1.0], dtype=jnp.float32), jnp.array([1.0, 1.0], dtype=jnp.float32)],
        coin_coop=[jnp.array([1, 2], dtype=jnp.int8), 
                jnp.array([2, 1], dtype=jnp.int8),
                jnp.array([1, 5], dtype=jnp.int8),
                jnp.array([2, 6], dtype=jnp.int8),
                jnp.array([6, 2], dtype=jnp.int8),
                jnp.array([5, 1], dtype=jnp.int8),],
        coin_defect=[jnp.array([1, 1], dtype=jnp.int8), 
                jnp.array([2, 2], dtype=jnp.int8),
                jnp.array([1, 6], dtype=jnp.int8),
                jnp.array([2, 5], dtype=jnp.int8),
                jnp.array([6, 1], dtype=jnp.int8),
                jnp.array([5, 2], dtype=jnp.int8),],
        agent_freezes=[jnp.array(-1), jnp.array(-1)],
    )
    actions = [jnp.array(2), jnp.array(2)]
    # a1 = 2
    # a2 = 2

    obs, new_state, _, _, _ = env.step(rng, state, actions, params)
    assert (new_state.red_pos[:2] != new_state.blue_pos[:2]).any()
