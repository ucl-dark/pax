import os
import time

import jax
from pytest import mark

from pax.envs.rice import Rice, EnvParams

file_dir = os.path.join(os.path.dirname(__file__))
env_dir = os.path.join(file_dir, "../../pax/envs")

def test_rice():
    rng = jax.random.PRNGKey(0)
    ep_length = 20
    num_players = 27

    env = Rice(num_inner_steps=ep_length, config_folder=os.path.join(env_dir, "region_yamls"))
    env_params = EnvParams()
    obs, env_state = env.reset(rng, EnvParams())

    for i in range(3 * ep_length + 1):
        # Do random actions
        key, _ = jax.random.split(rng, 2)
        action = jax.random.uniform(rng, (env.num_actions,))
        actions = tuple([action for _ in range(num_players)])
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, actions, env_params
        )
        for i in range(num_players):
            assert env_state.consumption_all[i].item() >= 0, "consumption cannot be negative!"

        # assert all obs positive
        # assert done firing correctly


# TODO assert gross_output - investment - total_exports > -1e-5, "consumption cannot be negative!"

@mark.skip(reason="Benchmark")
def test_benchmark_rice_n():
    rng = jax.random.PRNGKey(0)
    ep_length = 100
    iterations = 100
    num_players = 27

    env = Rice(num_inner_steps=ep_length, config_folder=os.path.join(env_dir, "region_yamls"))
    env_params = EnvParams()
    obs, env_state = env.reset(rng, EnvParams())

    start_time = time.time()

    for i in range(iterations * ep_length):
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
    print(f"Total time taken:\t{total_time:.4f} seconds")
    print(f"Average step duration:\t{total_time / (iterations * ep_length):.4f} seconds")
