# Pax
Pax is a JAX Batched Environment for other agent shaping. It supports both regular and meta agents, and both evolutionary strategies and RL based optimisation strategies.

> *Pax (noun) - a period of peace that has been forced on a large area, such as an empire or even the whole world*

PAX has a simple interface, similar to [dm-env](https://github.com/deepmind/dm_env). The API has vairants of traditional gym calls: `runner_step` and `runner_reset` which are compatible with `jit`, `vmap`, `pmap` and `lax.scan`.

```python
import SequentialMatrixGame

env = SequentialMatrixGame(
     num_envs=1,
     payoff=[[-1, -1], [0, -3], [-3, 0], [-2, -2]],
     inner_ep_length=10,
     num_steps=10
)

timesteps, env_state = env.reset()
agents = IndependentLearners(
     agent_0,
     agent_1
)

while not timestep[0].last():
     actions = agents.step(timesteps, )
     timestep, env_state = env.step(actions, env_state)
```

and timestep returns the following:

```python
timestep = timesteps[0]
timestep.observation.shape()
# (num_envs, num_states)
timestep.reward.shape()
# (num_envs, )
```
*Note: The original form of Pax was much closer to Acme but diverges further every day as we vmap, scan, and JIT more components for faster training. 

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

We currently support two major environments: `MatrixGames` and `CoinGame`. By default, PAX plays the [Iterated Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma) but we also include three other common matrix games: [Stag Hunt](https://en.wikipedia.org/wiki/Stag_hunt), [Battle of the Sexes](https://en.wikipedia.org/wiki/Battle_of_the_sexes_(game_theory)), and [Chicken](https://en.wikipedia.org/wiki/Chicken_(game)). The payoff matrices are as follows: 
```     
            CC        CD       DC       DD
IPD       = [[-2,-2], [-3, 0], [ 0,-3], [-1,-1]]
Stag hunt = [[ 4, 4], [ 3, 1], [ 1, 3], [ 2, 2]]
BotS      = [[ 3, 2], [ 0, 0], [ 0, 0], [ 2, 3]]
Chicken   = [[ 0, 0], [ 1,-1], [-1, 1], [-2,-2]]
``` 

These games can be called using the following flags: ```ipd, stag, sexes, chicken``` or by editing the `yaml` file. For example: 

```bash 
python -m pax.experiment +experiment/ipd=ppo ++game="ipd" ++agent2=TitForTat,Defect ++wandb.group="testing" --multirun
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
PAX includes a number of learning and fixed agents. They are specified in the `.yaml` files as `Agent1` and `Agent2`. Canonically, we care about the outcome of `Agent1`. All of the learning strategies have their own folder and the fixed agents can be viewed in `pax/strategies.py` (inspired by Axelrods' Tournament). 

| Agents | Description |
| -------| ----------- |
| `Naive`      | A learning agent that updates using REINFORCE [insert paper here]         |
| `PPO`    | A learning agent that updates using PPO [insert paper here]   |
| `PPO_memory`    | A learning agent that updates using PPO and a memory state [insert paper here]       |
| `MFOS`    | A meta learning agent that updates using PPO [insert paper here]       |
| `Defect`    | A fixed agent that always defects        |
| `Altruistic`    | A fixed agent that always cooperates        |
| `TitForTat`    | A fixed agent that cooperates on the first move and then reciprocates action of the opponent from the previous turn    |

*Note: MFOS is a meta-reinforcement learning algorithm, so will only work with the meta environment

## Environments 
PAX includes three environments specified by `env_id`. These are `ipd` and `coin_game`. Independetly you can specify your enviroment type by `env_type`, the options supported are `sequential`, `meta` and `infinite`. These are specified in the config files in `pax/configs/{env_id}/EXPERIMENT.yaml`. 

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
1. All models trained using PAX by default are saved to the `exp` folder. 
2a. If you have the model saved locally, specify `model_path = exp/...`. By default, Player 1 will be loaded with the parameters.  
2b. If you do not have the weights saved locally, specify the wandb run `run_path={wandb-group}{wandb-project}{}` and `model_path = exp/...` player 1 will be loaded with the parameters. 
3. In order to run evaluation, specify `eval: True` and evaluation for `num_seeds` iterations. 
