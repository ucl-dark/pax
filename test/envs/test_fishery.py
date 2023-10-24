import jax
import jax.numpy as jnp

from pax.envs.fishery import EnvParams, Fishery


def test_fishery_convergence():
    rng = jax.random.PRNGKey(0)
    ep_length = 300

    env = Fishery(num_players=2)
    env_params = EnvParams(g=0.15, e=0.009, P=200, w=0.9, s_0=1.0, s_max=1.0)
    # response parameter

    obs, env_state = env.reset(rng, env_params)
    d = 0.4
    E = 1.0
    step_reward = 0
    for i in range(3 * ep_length + 1):
        if i != 0 and env_state.s > 0:
            E = E + d * step_reward

        obs, env_state, rewards, done, info = env.step(
            rng, env_state, (E / 2, E / 2), env_params
        )
        step_reward = rewards[0] + rewards[1]

        # Check convergence at the end of an episode
        if i % ep_length == ep_length - 2:
            S_star = env_params.w / (
                env_params.P * env_params.e * env_params.s_max
            )  # 0.5
            assert jnp.isclose(S_star, env_state.s, atol=0.01)

            H_star = (
                env_params.w * env_params.g / (env_params.P * env_params.e)
            ) * (
                1 - S_star
            )  # 0.0375
            assert jnp.isclose(H_star, info["H"], atol=0.01)

            E_star = (env_params.g / env_params.e) * (1 - S_star)  # ~ 8.3333
            assert jnp.isclose(E_star, E, atol=0.01)

        # Check that the environment resets correctly
        if done is True:
            assert env_state.inner_t == 0
            assert env_state.outer_t == i // ep_length
            assert env_state.s == env_params.s_0
            assert step_reward == 0
