import os
import time

import jax

from pax.envs.rice.rice import Rice, EnvParams

file_dir = os.path.join(os.path.dirname(__file__))
config_folder = os.path.join(file_dir, "../../pax/envs/rice/5_regions")
num_players = 5
ep_length = 20


def test_rice():
    rng = jax.random.PRNGKey(0)

    env = Rice(config_folder=config_folder, episode_length=ep_length)
    env_params = EnvParams()
    obs, env_state = env.reset(rng, EnvParams())

    key, _ = jax.random.split(rng, 2)
    for i in range(100 * ep_length + 1):
        # Do random actions
        key, _ = jax.random.split(key, 2)
        actions = jax.random.uniform(key, (num_players, env.num_actions))
        actions = tuple([action for action in actions])
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, actions, env_params
        )
        for j in range(num_players):
            # assert all obs positive
            assert (
                env_state.consumption_all[j].item() >= 0
            ), "consumption cannot be negative!"
            assert (
                env_state.production_all[j].item() >= 0
            ), "production cannot be negative!"
            assert (
                env_state.labor_all[j].item() >= 0
            ), "labor cannot be negative!"
            assert (
                env_state.capital_all[j].item() >= 0
            ), "capital cannot be negative!"
            assert (
                env_state.gross_output_all[j].item() >= 0
            ), "gross output cannot be negative!"
            assert (
                env_state.investment_all[j].item() >= 0
            ), "investment cannot be negative!"

        if done is True:
            assert env_state.inner_t == 0
            assert (i + 1) / ep_length == 0
        assert (
            i + 1
        ) % ep_length == env_state.inner_t, "inner_t not updating correctly"


def rice_performance_benchmark():
    rng = jax.random.PRNGKey(0)
    iterations = 1000

    env = Rice(config_folder=config_folder, episode_length=ep_length)
    env_params = EnvParams()
    obs, env_state = env.reset(rng, EnvParams())

    start_time = time.time()

    for _ in range(iterations * ep_length):
        # Do random actions
        key, _ = jax.random.split(rng, 2)
        action = jax.random.uniform(rng, (env.num_actions,))
        actions = tuple([action for _ in range(num_players)])
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, actions, env_params
        )

    end_time = time.time()  # End timing
    total_time = end_time - start_time

    # Print or log the total time taken for all iterations
    print(f"Total iterations:\t{iterations * ep_length}")
    print(f"Total time taken:\t{total_time:.4f} seconds")
    print(
        f"Average step duration:\t{total_time / (iterations * ep_length):.6f} seconds"
    )


# Run a benchmark
if __name__ == "__main__":
    rice_performance_benchmark()
