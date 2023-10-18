import os

import jax
import jax.numpy as jnp
import jaxopt
import optax

from pax.envs.rice.rice import EnvParams
from pax.envs.rice.sarl_rice import SarlRice
from jaxopt import ProjectedGradient, OptaxSolver
from jaxopt.projection import projection_non_negative

jax.config.update("jax_enable_x64", True)


ep_length = 20
env_dir = os.path.join("./pax/envs/rice")
sarl_env = SarlRice(config_folder=os.path.join(env_dir, "5_regions"), episode_length=ep_length)
env_params = EnvParams()


def objective(params, rng):
    obs, state = sarl_env.reset(rng, env_params)
    rewards = 0
    for i in range(ep_length):
        obs, state, reward, done, info = sarl_env.step(rng, state, params[i], env_params)
        rewards += jnp.asarray(reward).sum()

    return rewards


rng = jax.random.PRNGKey(0)

w_init = jnp.ones((ep_length, sarl_env.num_actions)) * 0.5

opt = optax.adam(0.0001)
solver = OptaxSolver(opt=opt, fun=objective, maxiter=100)
# solver = jaxopt.LBFGS(fun=objective, maxiter=100)
# pg = ProjectedGradient(fun=objective, projection=projection_non_negative)
pg_sol = solver.run(w_init, rng=rng)
params, state = pg_sol
print(pg_sol)
final_reward = objective(params, rng)
print(state)
print(f"reward {final_reward}")
