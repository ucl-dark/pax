import jax
import jax.numpy as jnp

from pax.envs.cournot import EnvParams, CournotGame


def test_single_cournot_game():
    rng = jax.random.PRNGKey(0)

    env = CournotGame(num_steps=1)
    # This means the optimum production quantity is Q = q1 + q2 = 2(a-marginal_cost)/3b = 60
    env_params = EnvParams(a=100, b=1, marginal_cost=10)
    opt_policy = 30 * jnp.ones((1))

    obs, env_state = env.reset(rng, env_params)
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, (opt_policy, opt_policy), env_params
    )


    assert rewards[0] == rewards[1]
    # p_opt = 100 - (30 + 30) = 40
    # r1_opt = 40 * 30 - 10 * 30 = 900
    assert jnp.isclose(900, rewards[0], atol=0.01)
    assert jnp.allclose(obs[0], jnp.array([30, 40]), atol=0.01)
    assert jnp.allclose(obs[1], jnp.array([30, 40]), atol=0.01)

    # TODO implement and test overproduction

