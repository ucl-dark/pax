import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import tree
from jax.config import config

# config.update('jax_disable_jit', True)

from pax.envs.rice.rice import Rice, EnvParams

file_dir = os.path.join(os.path.dirname(__file__))
config_folder = os.path.join(file_dir, "../../pax/envs/rice/5_regions")
num_players = 5
ep_length = 20


def test_rice():
    rng = jax.random.PRNGKey(0)

    env = Rice(num_inner_steps=ep_length, config_folder=config_folder)
    env_params = EnvParams()
    obs, env_state = env.reset(rng, EnvParams())

    for i in range(3 * ep_length + 1):
        # Do random actions
        key, _ = jax.ranydom.split(rng, 2)
        action = jax.random.uniform(rng, (env.num_actions,))
        actions = tuple([action for _ in range(num_players)])
        obs, env_state, rewards, done, info = env.step(
            rng, env_state, actions, env_params
        )
        for j in range(num_players):
            # assert all obs positive
            assert env_state.consumption_all[i].item() >= 0, "consumption cannot be negative!"

        if done is True:
            assert env_state.inner_t == 0
            assert (i + 1) / ep_length == 0
        assert (i + 1) % ep_length == env_state.inner_t, "inner_t not updating correctly"


def test_rice_regression(data_regression):
    rng = jax.random.PRNGKey(0)

    env = Rice(num_inner_steps=ep_length, config_folder=config_folder)  # Make sure to define 'config_folder'
    env_params = EnvParams()
    obs, env_state = env.reset(rng, env_params)

    results = {}

    for i in range(ep_length):
        action = jnp.ones((env.num_actions,)) * 0.5  # Deterministic action for testing
        actions = tuple([action for _ in range(num_players)])
        obs, env_state, rewards, done, info = env.step(rng, env_state, actions, env_params)

        stored_obs = [_obs.round(1).tolist() for _obs in obs]
        stored_env_state = [leaf.round(1).tolist() for leaf in tree.flatten(env_state)]
        stored_rewards = [reward.round(1).item() for reward in rewards]
        results[i] = {
            "obs": stored_obs,
            "env_state": stored_env_state,
            "rewards": stored_rewards,
        }

    data_regression.check(results)


def rice_performance_benchmark():
    rng = jax.random.PRNGKey(0)
    iterations = 1000

    env = Rice(num_inner_steps=ep_length, config_folder=config_folder)
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
    print(f"Total iterations:\t{iterations * ep_length}")
    print(f"Total time taken:\t{total_time:.4f} seconds")
    print(f"Average step duration:\t{total_time / (iterations * ep_length):.6f} seconds")


# Run a benchmark
if __name__ == "__main__":
    rice_performance_benchmark()
