<h1 align="center">
  <a href="https://github.com/akbir/pax/blob/main/docs/imgs/logo.png">
    <img src="https://github.com/akbir/pax/blob/main/docs/imgs/logo.png?raw=true" width="215" /></a><br>
  <b> Pax: Meta|Multi Agent Learning in JAX </b><br>
</h1>
<p align="center">
      <a href="https://pypi.python.org/pypi/gymnax">
        <img src="https://img.shields.io/badge/python-3.9-blue.svg" /></a>
       <a href= "https://github.com/akbir/pax/blob/main/LICENSE.md">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
       <a href= "https://codecov.io/gh/akbir/pax">
        <img src="https://codecov.io/gh/akbir/gymnax/branch/main/graph/badge.svg?token=OKKPDRIQJR" /></a>
       <a href= "https://github.com/psf/black">
        <img src="https://img.shields.io/badge/code%20style-black-000000.svg" /></a>
</p>


Pax is an experiment runner for multi and meta agent research built on top of JAX. It supports "other agent shaping", "multi agent RL" and "single agent RL" experiments. It supports evolutionary and RL-based optimisation. 

> *Pax (noun) - a period of peace that has been forced on a large area, such as an empire or even the whole world*

Pax is composed of 3 components: Environments, Agents and Runners.

### Environments
Environments are similar to [gymnax](https://github.com/RobertTLange/gymnax).

```python
from pax.envs.iterated_matrix_game import (
    IteratedMatrixGame,
    EnvParams,
)

env = IteratedMatrixGame(num_inner_steps=5)
env_params = EnvParams(payoff_matrix=payoff)

# 0 = Defect, 1 = Cooperate
actions = (jnp.ones(()), jnp.ones(()))
obs, env_state = env.reset(rng, env_params)
done = False

while not done:
    obs, env_state, rewards, done, info = env.step(
        rng, env_state, actions, env_params
    )
```
We can compose these with JAX built-in functins `jit`, `vmap`, `pmap` and `lax.scan`.

```python
import IteratedMatrixGame, EnvParams
import jax.numpy as jnp

# batch over env initalisations
num_envs = 2
payoff = [[2, 2], [0, 3], [3, 0], [1, 1]]
rollout_length = 50

rng = jnp.concatenate(
    [jax.random.PRNGKey(0), jax.random.PRNGKey(1)]
).reshape(num_envs, -1)

env = IteratedMatrixGame(num_inner_steps=rollout_length)
env_params = EnvParams(payoff_matrix=payoff)

action = jnp.ones((num_envs,), dtype=jnp.float32)

# we want to batch over rngs, actions
env.step = jax.vmap(
    env.step,
    in_axes=(0, None, 0, None),
    out_axes=(0, None, 0, 0, 0),
)

obs, env_state = env.reset(rng, env_params)

# lets scan the rollout for speed
def rollout(carry, unused):
    carry = (_, env_state, env_rng)
    actions = (action, action)
    obs, env_state, rewards, done, info = env.step(
        env_rng, env_state, actions, env_params
    )

    return (obs, env_state, env_rng), (
        obs,
        actions,
        rewards,
        done,
    )


final_state, trajectory = jax.lax.scan(
    rollout, (obs, env_state, rng), rollout_length
)
```

### Agents
The agent interface is as follows:

```python
import jax.numpy as jnp
import Agent

args = {"hidden": 16, "observation_spec": 5}
rng = jax.random.PRNGKey(0)
bs = 1
init_hidden = jnp.zeros((bs, args.hidden))
obs = jnp.ones((bs, 5))

agent = Agent(args)
state, memory = agent.make_initial_state(rng, init_hidden)
action, state, mem = agent.policy(rng, obs, mem)

state, memory, stats = agent.update(
    traj_batch, obs, state, mem
)

mem = agent.reset_memory(mem, False)
```

Note that `make_initial_state`, `policy`, `update` and `reset_memory` all support `jit`, `vmap` and `lax.scan`. Allowing you to compile more of your experiment to [XLA](https://www.tensorflow.org/xla).


```python
# batch MemoryState not TrainingState
agent.batch_reset = jax.jit(
    jax.vmap(agent.reset_memory, (0, None), 0),
    static_argnums=1,
)

agent.batch_policy = jax.jit(
    jax.vmap(agent._policy, (None, 0, 0), (0, None, 0))
)
agent1.batch_init = jax.vmap(
    agent.make_initial_state,
    (None, 0),
    (None, 0),
)
```

### Runners
We can finally combine all the above into our runner code. This is where you'd expect to write most custom logic for your own experimental set up,

```python
def _rollout(carry, unused):
    """Runner for inner episode"""
    (
        rngs,
        obs,
        a_state,
        a_mem,
        env_state,
        env_params,
    ) = carry

    # unpack rngs
    rngs = self.split(rngs, 4)
    action, a_state, new_a_mem = agent1.batch_policy(
        a_state,
        obs[0],
        a_mem,
    )

    next_obs, env_state, rewards, done, info = env.step(
        rngs,
        env_state,
        (action, action),
        env_params,
    )

    traj = Sample(
        obs1,
        action,
        rewards[0],
        new_a1_mem.extras["log_probs"],
        new_a1_mem.extras["values"],
        done,
        a1_mem.hidden,
    )

    return (
        rngs,
        next_obs,
        a1_state,
        new_a1_mem,
        env_state,
        env_params,
    ), (
        traj1,
        traj2,
    )


agent = Agent(args)
state, memory = agent.make_initial_state(rng, init_hidden)

for _ in range(num_updates):
    final_timestep, batch_trajectory = jax.lax.scan(
        _rollout,
        ((obs, env_state, rng), rollout_length),
        10,
    )

    _, obs, rewards, a1_state, a1_mem, _, _ = final_timestep

    state, memory, stats = agent.update(
        batch_trajectory, obs[0], state, memory
    )
```

Note this isn't even a fully optimised example - we could jit the outer loop!


# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to install JAX with the relevant accelerator support.

## General Information
The project entrypoint is `pax/experiment.py`. The simplest command to run a game would be: 

```bash
python -m pax.experiment
```

We currently use [WandB](https://wandb.ai/) for logging and [Hydra](https://hydra.cc/docs) for configs. Hyperparameters are stored `/conf/experiment` as `.yaml` files. Depending on your needs, you can specify hyperparameters through the CLI or by changing the `.yaml` files directly. 

```bash
python -m pax.experiment +total_timesteps=1_000_000 +num_envs=10
```

We currently support two major environments: `MatrixGames` and `CoinGame`.
```

For `MatrixGames`, we support the ability to specify your own payoff matrix either through the CLI or the `yaml` files. For example the common Iterated Prisoners Dilemma is:
```bash 
python -m pax.experiment +experiment/ipd=ppo ++payoff="[[-2,-2], [0,-3], [-3,0], [-1,-1]]" ++wandb.group="testing"
```

## Experiments
```bash 
python -m pax.experiment +experiment/ipd=yaml ++wandb.group="testing" 
``` 

We store previous experiments as parity tests. We use [Hydra](https://hydra.cc/docs) to store these configs and keep track of good hyper-paremters. As a rule for development, we try retain backwards compatability and allow all previous results to be replicated. These can be run easily by `python -m pax.experiment +experiment=NAME`. We also provide a  list of our existing experiments and expected result [here](docs/experiments.md).

## Loading and Saving 
1. All models trained using Pax by default are saved to the `exp` folder. 
2a. If you have the model saved locally, specify `model_path = exp/...`. By default, Player 1 will be loaded with the parameters.  
2b. If you do not have the weights saved locally, specify the wandb run `run_path={wandb-group}{wandb-project}{}` and `model_path = exp/...` player 1 will be loaded with the parameters. 
3. In order to run evaluation, specify `eval: True` and evaluation for `num_seeds` iterations. 


## Citation

If you use Pax in any of your work, please cite:

```
@misc{pax,
    author = {Khan, Akbir and Willi, Timon and Kwan, Newton, and Samvelyan, Mikayel and Lu, Chris},
    title = {Pax: Multi-Agent Learning in JAX},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/akbir/pax}},
}
```