import jax
import jax.numpy as jnp

from pax.envs.cournot import EnvParams, CournotGame


def test_single_cournot_game():
    rng = jax.random.PRNGKey(0)

    for n_player in [2, 3, 12]:
        env = CournotGame(num_players=n_player, num_inner_steps=1)
        # This means the optimum production quantity is Q = q1 + q2 = 2(a-marginal_cost)/3b = 60
        env_params = EnvParams(a=100, b=1, marginal_cost=10)
        nash_q = CournotGame.nash_policy(env_params)
        assert nash_q == 60
        nash_action = jnp.array([nash_q / n_player])

        obs, env_state = env.reset(rng, env_params)
        obs, env_state, rewards, done, info = env.step(
            rng,
            env_state,
            tuple([nash_action for _ in range(n_player)]),
            env_params,
        )

        assert all(element == rewards[0] for element in rewards)
        # p_opt = 100 - (30 + 30) = 40
        # r1_opt = 40 * 30 - 10 * 30 = 900
        nash_reward = CournotGame.nash_reward(env_params)
        assert nash_reward == 1800
        assert jnp.isclose(nash_reward / n_player, rewards[0], atol=0.01)
        expected_obs = jnp.array(
            [60 / n_player for _ in range(n_player)] + [40]
        )
        assert jnp.allclose(obs[0], expected_obs, atol=0.01)
        assert jnp.allclose(obs[0], obs[1], atol=0.0)

        social_opt_action = jnp.array([45 / n_player])
        obs, env_state = env.reset(rng, env_params)
        obs, env_state, rewards, done, info = env.step(
            rng,
            env_state,
            tuple([social_opt_action for _ in range(n_player)]),
            env_params,
        )
        assert jnp.asarray(rewards).sum() == 2025
