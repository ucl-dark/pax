<h1 align="center">
  <a href="https://github.com/akbir/pax/blob/main/docs/logo.png">
    <img src="https://github.com/akbir/pax/blob/main/docs/logo.png?raw=true" width="215" /></a><br>
  <b> Pax: Multi-Agent Learning in JAX </b><br>
</h1>


Pax is an experiment runner for multi-agent research built on top of JAX. It supports "other agent shaping", "multi agent RL" and "single agent RL" experiments. It supports regular and meta agents, and evolutionary and RL-based optimisation. 

> *Pax (noun) - a period of peace that has been forced on a large area, such as an empire or even the whole world*

Pax is composed of 3 components: Environments, Agents and Runners.

### Environments
Environments are similar to [gymnax](https://github.com/RobertTLange/gymnax).

```python
import IteratedMatrixGame, EnvParams

env = IteratedMatrixGame(num_inner_steps=5)
env_params = EnvParams(payoff_matrix=payoff)

# 0 = Defect, 1 = Cooperate
actions = (jnp.ones(()), jnp.ones(()))
obs, env_state = env.reset(rng, env_params)
done = False

while not done:
     obs, env_state, rewards, done, info = env.step(
          rng,
          env_state,
          actions,
          env_params)
```
Similar to [gymnax](https://github.com/RobertTLange/gymnax), we can compose these with JAX built-in functins `jit`, `vmap`, `pmap` and `lax.scan`.

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
r_array = jnp.ones((num_envs,), dtype=jnp.float32)

# we want to batch over envs purely by actions
env.step = jax.vmap(
     env.step, in_axes=(0, None, 0, None), out_axes=(0, None, 0, 0, 0)
)
obs, env_state = env.reset(rng, env_params)

# lets scan the rollout for speed
def rollout(carry, unused):
     actions = (action, action)
     carry = (_, env_state, env_rng)

     obs, env_state, rewards, done, info = env.step(
          env_rng,
          env_state,
          actions,
          env_params)
     
     return (obs, env_state, env_rng) (obs, actions, rewards, done)

final_state, trajectory = jax.lax.scan(
     rollout, (obs, env_state, rng), rollout_length)
```

### Agents
The agent interface is as follows:

```python
import jax.numpy as jnp
import Agent

args = {'hidden'= 16, 'observation_spec'=5}
rng = jax.random.PRNGKey(0)
bs = 1
init_hidden = jnp.zeros((bs, args.hidden))
obs = jnp.ones((bs, 5))

agent = Agent(args)
state, memory =  agent.make_initial_state(rng, init_hidden)
action, state, mem = agent.policy(rng, obs, mem)

state, memory, stats = agent.update(
     traj_batch,
     obs,
     state,
     mem)

mem = agent.reset_memory(mem, False)
```

Note that `make_initial_state`, `policy`, `update` and `reset_memory` all support `jit`, `vmap` and `lax.scan`. Allowing you to compile more of your experiment to [XLA](https://www.tensorflow.org/xla).


```python
     # batch MemoryState not TrainingState
     agent.batch_reset = jax.jit(
          jax.vmap(agent.reset_memory, (0, None), 0), static_argnums=1
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
               r,
               a1_state,
               a1_mem,
               env_state,
               env_params,
          ) = carry

          # unpack rngs
          rngs = self.split(rngs, 4)
          a1, a1_state, new_a1_mem = agent1.batch_policy(
               a1_state,
               obs[0],
               a1_mem,
          )

          next_obs, env_state, rewards, done, info = env.step(
               rngs,
               env_state,
               (a1, a1),
               env_params,
          )

          traj = Sample(
               obs1,
               a1,
               rewards[0],
               new_a1_mem.extras["log_probs"],
               new_a1_mem.extras["values"],
               done,
               a1_mem.hidden,
               )

          return (
               rngs,
               next_obs,
               rewards,
               a1_state,
               new_a1_mem,
               env_state,
               env_params,
          ), (
               traj1,
               traj2,
          )


     agent = Agent(args)
     state, memory =  agent.make_initial_state(rng, init_hidden)
     
     for _ in range(num_updates):
          final_timestep, batch_trajectory = jax.lax.scan(
               _rollout,
               ((obs, env_state, rng), rollout_length),
               10
          )

          _, obs, rewards, a1_state, a1_mem, _, _ = final_timestep

          state, memory, stats = agent.update(
               batch_trajectory,
               obs[0],
               rewards[0],
               state,
               memory
           )
```

Note this isn't even a fully optimised example - we could jit the outer loop!


# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to install JAX with the relevant accelerator support.

# Getting Started
## General Information
The project entrypoint is `pax/experiment.py`. The simplest command to run a game would be: 

```bash
python -m pax.experiment
```

We currently use [WandB](https://wandb.ai/) for logging and [Hydra](https://hydra.cc/docs) for configs. Hyperparameters are stored `/conf/experiment` as `.yaml` files. Depending on your needs, you can specify hyperparameters through the CLI or by changing the `.yaml` files directly. 

```bash
python -m pax.experiment +total_timesteps=1_000_000 +num_envs=10
```

We currently support two major environments: `MatrixGames` and `CoinGame`. By default, Pax plays the [Iterated Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma) but we also include three other common matrix games: [Stag Hunt](https://en.wikipedia.org/wiki/Stag_hunt), [Battle of the Sexes](https://en.wikipedia.org/wiki/Battle_of_the_sexes_(game_theory)), and [Chicken](https://en.wikipedia.org/wiki/Chicken_(game)). The payoff matrices are as follows: 
```     
            CC        CD       DC       DD
IPD       = [[-2,-2], [-3, 0], [ 0,-3], [-1,-1]]
Stag hunt = [[ 4, 4], [ 3, 1], [ 1, 3], [ 2, 2]]
BotS      = [[ 3, 2], [ 0, 0], [ 0, 0], [ 2, 3]]
Chicken   = [[ 0, 0], [ 1,-1], [-1, 1], [-2,-2]]
``` 

These games can be called using the following flags: ```ipd, stag, sexes, chicken``` or by editing the `yaml` file. For example: 

```bash 
python -m pax.experiment +experiment/ipd=ppo ++game="matrix_game" ++agent2=TitForTat,Defect ++wandb.group="testing" --multirun
``` 
or 

```yaml
# Config.yaml file
...
game: ipd
...
```



Additionally, we support the ability to specify your own payoff matrix either through the CLI or the `yaml` files: 

```bash 
python -m pax.experiment +experiment/ipd=ppo ++payoff="[[-2,-2], [0,-3], [-3,0], [-1,-1]]" ++wandb.group="testing"
```

## Agents
Pax includes a number of learning and fixed agents. They are specified in the `.yaml` files as `Agent1` and `Agent2`. Canonically, we care about the outcome of `Agent1`. All of the learning strategies have their own folder and the fixed agents can be viewed in `pax/strategies.py` (inspired by Axelrods' Tournament). 

| Agents | Description |
| -------| ----------- |
| `Naive`      | A learning agent that updates using REINFORCE [insert paper here]         |
| `PPO`    | A learning agent that updates using PPO [insert paper here]   |
| `PPO_memory`    | A learning agent that updates using PPO and a memory state [insert paper here]       |
| `MFOS`    | Model free opponent shaping meta-agent based upon the [ICML paper 2021](https://arxiv.org/abs/2205.01447)     |
| `GS`    | The Good Shepard meta-agent based upon the [paper](https://arxiv.org/abs/2202.10135)|
| `Defect`    | A fixed agent that always defects        |
| `Altruistic`    | A fixed agent that always cooperates        |
| `TitForTat`    | A fixed agent that cooperates on the first move and then reciprocates action of the opponent from the previous turn    |

*Note: `MFOS` and `GS` are meta-agents, so will only work with the meta environment

## Environments 
Pax includes three environments specified by `env_id`. These are `ipd` and `coin_game`. Independetly you can specify your enviroment type by `env_type`, the options supported are `sequential`, `meta` and `infinite`. These are specified in the config files in `pax/configs/{env_id}/EXPERIMENT.yaml`. 

| Environment ID | Environment Type | Description |
| ----------- | ----------- | ----------- |
|ipd| `sequential`      | An iterated matrix game with a predetermined number of timesteps per episode with a discount factor $\gamma$         |
|ipd | `infinite`    | An infinite matrix game that calculates exact returns given a payoff and discount factor $\gamma$       |
|ipd | `meta`    | A meta game over the iterated matrix game with an outer agent (player 1) and an inner agent (player 2). The inner updates every episode, while the the outer agent updates every meta-episode |
|coin_game | `sequential`    | A sequential series of episode of the coin game between two players. Each player updates at the end of an episode|
|coin_game | `meta`    | A meta learning version of the coin game with an outer agent (player 1) and an inner agent (player 2). The inner updates every episode, while the the outer agent updates every meta-episode|

## Experiments
```bash 
python -m pax.experiment +experiment/ipd=yaml ++wandb.group="testing" 
``` 

We store previous experiments as parity tests. We use [Hydra](https://hydra.cc/docs) to store these configs and keep track of good hyper-paremters. As a rule for development, we try retain backwards compatability and allow all previous results to be replicated. These can be run easily by `python -m pax.experiment +experiment=NAME` Below are a list of our existing experiments and expected result. Canonically, the `Agent 1` agent type is constant in each experiment while `Agent 2` changes. 

| yaml | Agent1 | Agent2 | Environment | Outcome |
| ----------- | ------- | ------- | --- | ----------- |
| `ipd/naive`| `Naive`| `Any` | `Finite` | `Agent2: TitForTat -> ALL-C; [Naive, Defect, Altruistic] -> ALL-D` |
| `ipd/PPO`| `PPO`| `Any`| `Finite` | `Agent2: TitForTat -> ALL-C; [PPO, Defect, Altruistic] -> ALL-D` |
| `ipd/PPO_memory`| `PPO_memory`| `Any`| `Finite` | `Agent2: TitForTat -> ALL-C; [PPO_memory, Defect, Altruistic] -> ALL-D`  |
| `ipd/mfos_fixed`  | `PPO`| `Fixed` | `Infinite` |`Agent2: TitForTat -> ALL-C; [Defect, Altruistic] -> ALL-D` |
|  `ipd/mfos_nl`    | `PPO`| `Naive` |  `Infinite` |`Agent2: Naive -> ZD-Extortion (Player 1 has a payoff greater than CC)`  |
| `ipd/marl2_fixed` | `PPO_memory` | `Fixed` | `Meta`| `Agent2: TitForTat -> ALL-C; [Defect, Altruistic] -> ALL-D` |
| `ipd/marl2_nl`    | `PPO_memory` | `Naive` | `Meta` |`Agent2: Naive -> Mixture of Cooperation and Defection` |
| `ipd/earl_fixed`| `PPO_memory`| `Fixed`| `Meta` | `Agent2: TitForTat -> ALL-C; [Defect, Altruistic] -> ALL-D` |
| `ipd/earl_nl_cma`| `PPO_memory`| `Naive`| `Meta` | `Agent2: Naive -> ZD-Extortion`|
| `ipd/earl_nl_open`| `PPO_memory`| `Naive`| `Meta` | `Agent2: Naive -> ZD-Extortion` |
| `ipd/earl_nl_pgpe`| `PPO_memory`| `Naive`| `Meta` | `Agent2: Naive -> TBD` |

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