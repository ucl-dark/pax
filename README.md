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

For a simple starting points, please refer to this [notebook](https://github.com/ucl-dark/pax/blob/main/proof_of_concept.ipynb), which introduces the very basic concepts of pax. Please read the README, the 5 minutes spent reading will save you a lot of time. If you run into any problems, please open an issue.

Pax is composed of 3 components: Environments, Agents and Runners.

### Environments
Environments are similar to [gymnax](https://github.com/RobertTLange/gymnax). We can compose these with JAX built-in functins `jit`, `vmap`, `pmap` and `lax.scan`. In the following example, we batch over the number of environments.

```python
from pax.envs.iterated_matrix_game import (
    IteratedMatrixGame,
    EnvParams,
)

import jax
import jax.numpy as jnp

num_envs = 2
payoff = [[2, 2], [0, 3], [3, 0], [1, 1]] # payoff matrix for the IPD
rollout_length = 50

rng = jnp.concatenate(
    [jax.random.PRNGKey(0), jax.random.PRNGKey(1)]
).reshape(num_envs, -1)

env = IteratedMatrixGame(num_inner_steps=rollout_length, num_outer_steps=1)
env_params = EnvParams(payoff_matrix=payoff)

action = jnp.ones((num_envs,), dtype=jnp.float32)

# we want to batch over rngs, actions
env.step = jax.vmap(
    env.step,
    in_axes=(0, None, 0, None),
    out_axes=(0, None, 0, 0, 0),
)

# batch over env initalisations
env.reset = jax.vmap(
    env.reset, in_axes=(0, None), out_axes=(0, None))
obs, env_state = env.reset(rng, env_params)

# lets scan the rollout for speed
def rollout(carry, unused):
    last_obs, env_state, env_rng = carry
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
    rollout, (obs, env_state, rng), None, rollout_length
)
```

A description of most available environments is available [here](https://github.com/ucl-dark/pax/blob/main/docs/envs.md).
### Agents
`pax` supports a wide range on agents. To initialise agents, we need to define their arguments. This is typically done in the `.yaml` file defining the experiment. In the following code, we'll define in explicitly.

```python
from typing import NamedTuple

class EnvArgs(NamedTuple):
    env_id='iterated_matrix_game'
    runner='rl'
    num_envs=num_envs

class PPOArgs(NamedTuple):
    num_minibatches=10
    num_epochs=4
    gamma=0.96
    gae_lambda=0.95
    ppo_clipping_epsilon=0.2
    value_coeff=0.5
    clip_value=True
    max_gradient_norm=0.5
    anneal_entropy=True
    entropy_coeff_start=0.2
    entropy_coeff_horizon=5e5
    entropy_coeff_end=0.01
    lr_scheduling=True
    learning_rate=2.5e-2
    adam_epsilon=1e-5
    with_memory=False
    with_cnn=False
    hidden_size=16
```
After having defined the arguments, we're ready to initialize the agents. We also vmap over the reset, policy and update function. We do not vmap over the initialisation, as the initialisation already assumes that we are running over multiple environments. If you were to add additional batch dimensions, you'd want to vmap over the initialisation too. The vmapping of the agents is typically done in the runner file.

Note that `make_initial_state`, `policy`, `update` and `reset_memory` all support `jit`, `vmap` and `lax.scan`. Allowing you to compile more of your experiment to [XLA](https://www.tensorflow.org/xla).


```python
import jax.numpy as jnp
from pax.agents.ppo.ppo import make_agent

args = EnvArgs()
agent_args = PPOArgs()
agent = make_agent(args, 
    agent_args=agent_args,
    obs_spec=env.observation_space(env_params).n,
    action_spec=env.num_actions,
    seed=42,
    num_iterations=1e3,
    player_id=0,
    tabular=False,)


# batch MemoryState not TrainingState
agent.batch_reset = jax.jit(
    jax.vmap(agent.reset_memory, (0, None), 0),
    static_argnums=1,
)

agent.batch_policy = jax.jit(
    jax.vmap(agent._policy, (None, 0, 0), (0, None, 0))
)

agent.batch_update = jax.vmap(
    agent.update, (0, 0, None, 0), (None, 0, 0)
)
```
A list of most available agents is available [here](https://github.com/ucl-dark/pax/blob/main/docs/agents.md).
### Runners
We can finally combine all the above into our runner code. This is where you'd expect to write most custom logic for your own experimental set up. Next, we'll define a transition. The transitions is what we're stacking when we're gonna use jax.lax.scan.

```python
from typing import NamedTuple

class Sample(NamedTuple):
    """Object containing a batch of data"""

    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    behavior_log_probs: jnp.ndarray
    behavior_values: jnp.ndarray
    dones: jnp.ndarray
    hiddens: jnp.ndarray
```

Now let's define a very simple rollout. Note that the agent plays against itself in the IPD, which isn't particularly meaningful. If you had a second agent, you'd obviously want to fill in that code accordingly. Please refer to the many runners within the repo that already have that code in place.

```python
def _rollout(carry, unused):
    """Runner for inner episode"""
    (
        rng,
        obs,
        a_state,
        a_mem,
        env_state,
        env_params,
    ) = carry
    # unpack rngs
    rngs = jax.random.split(rng, num_envs+1)
    rngs, rng = rngs[:-1], rngs[-1]

    action, a_state, new_a_mem = agent.batch_policy(
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
        obs[0],
        action,
        rewards[0],
        new_a_mem.extras["log_probs"],
        new_a_mem.extras["values"],
        done,
        a_mem.hidden,
    )

    return (
        rng,
        next_obs,
        a_state,
        new_a_mem,
        env_state,
        env_params,
    ), traj
```

Note this isn't even a fully optimised example - we could jit the outer loop! Now we can put everything together and train the agent. This already covers almost everything there is to know in pax. All the runners are just modified versions of this notebook, where initialisations are centralized in experiments.py

```python
rng = jax.random.PRNGKey(42)
init_hidden = jnp.zeros((agent_args.hidden_size))
rng, _rng = jax.random.split(rng)
a_state, a_memory = agent.make_initial_state(_rng, init_hidden)
rngs = jax.random.split(rng, num_envs)
obs, env_state = env.reset(rngs, env_params)

for _ in range(10):
    carry =  (rng, obs, a_state, a_memory, env_state, env_params)
    final_timestep, batch_trajectory = jax.lax.scan(
        _rollout,
        carry,
        None,
        10,
    )

    rng, obs, a_state, a_memory, env_state, env_params = final_timestep

    a_state, a_memory, stats = agent.update(
        batch_trajectory, obs[0], a_state, a_memory
    )
```
A list of most available runners is available [here](https://github.com/ucl-dark/pax/blob/main/docs/getting-started/runners.md). 

# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to [install](https://github.com/google/jax#installation) JAX with the relevant accelerator support.

Then, install the following [requirements](https://github.com/ucl-dark/pax/blob/main/requirements.txt).

## General Information
The project entrypoint is `pax/experiment.py`. The simplest command to run a game would be: 

```bash
python -m pax.experiment
```

We currently use [WandB](https://wandb.ai/) for logging and [Hydra](https://hydra.cc/docs) for configs. Hyperparameters are stored `/conf/experiment` as `.yaml` files. Depending on your needs, you can specify hyperparameters through the CLI or by changing the `.yaml` files directly. 

```bash
python -m pax.experiment +train_iters=1e3 +num_envs=10
```

We currently support the following environments: `MatrixGames` (both infinite and finite versions), `CoinGame`, `InTheMatrix` (recently renamed to [STORM](https://arxiv.org/pdf/2311.10090.pdf), an n-player version of STORM, and `iterated_tensor_games` (n-player version of matrix games). These environments are great testing grounds for general-sum games. From the climate literature, we also support the `Cournot`, `Fishery`, `Rice-N` and `C-Rice-N` environments. For more information about the environments, please see [here](https://github.com/ucl-dark/pax/blob/main/docs/envs.md).

For most games, we support the ability to specify your own payoff matrix either through the CLI or the `.yaml` files. For example, if you want to run the experiment where PPO plays against a TitForTat agent in the common Iterated Prisoners Dilemma:

```bash 
python -m pax.experiment +experiment/ipd=ppo_v_tft ++payoff="[[-2,-2], [0,-3], [-3,0], [-1,-1]]" ++wandb.log=False
```

## Experiments
```bash 
python -m pax.experiment +experiment/ipd=yaml
``` 

We store previous experiments as parity tests. We use [Hydra](https://hydra.cc/docs) to store these configs and keep track of good hyper-paremters. As a rule for development, we try retain backwards compatability and allow all previous results to be replicated. These can be run easily by `python -m pax.experiment +experiment=NAME`. We also provide a list of our existing experiments and expected result [here](docs/experiments.md), though it's not exhaustive and some `.yaml` files not be up to date. Just leave us a message if you run into any problems.

## Loading and Saving 
1. All models trained using Pax by default are saved to the `exp` folder. 
2a. If you have the model saved locally, specify `model_path = exp/...`. By default, Player 1 will be loaded with the parameters.  
2b. If you do not have the weights saved locally, specify the wandb run `run_path={wandb-group}{wandb-project}{}` and `model_path = exp/...` player 1 will be loaded with the parameters. 
3. In order to run evaluation, specify `eval: True` and evaluation for `num_seeds` iterations. 


## Citation

If you use Pax in any of your work, please cite:

```
@misc{pax,
    author = {Willi, Timon, and Khan, Akbir and Kwan, Newton, and Samvelyan, Mikayel and Lu, Chris, and Foerster, Jakob},
    title = {Pax: Multi-Agent Learning in JAX},
    year = {2023},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/ucl-dark/pax}},
}
```
