# Agents 

## Overview 

Pax provides a number of fixed opponents and learning agents to train and train against. 

## Specifying an Agent

Pax comes installed with an `Agent` class and several predefined agents. To specify an agent, import the `Agent` class and specify the agent parameters. 

```
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

To run an experiment with a specific agent, use a pre-made `.yaml` file located in `conf/...` or create your own, and specify the agent. In the below example, `agent1` is a learning agent that learns via PPO and `agent2` is an agent that only chooses the Cooperate action. 

```
# Agents  
agent1: 'PPO'
agent2: 'Altruistic'

...
```

## List of Agents

```{note}
Fixed agents are game-specific, while learning agents like PPO can be used in both games. 
```

### Fixed

Matrix games:

|  Agent      |  Description   | 
| ----------- | ----------- |
| **`Altruistic`**  | Always chooses the Cooperate (C) action. |
| **`Defect`**     | Always chooses the Defect (D) action. |
| **`GrimTrigger`**   | Chooses the C action on the first turn and reciprocates with the C action until the opponent chooses D, where Grim switches to only choosing D.|
| **`HyperAltruistic`**  | Infinite matrix game variant of `Altruistic`. Always chooses the Cooperate (C) action.|
| **`HyperDefect`**  | Infinite matrix game variant of `Defect`. Always chooses the Defect (D) action.|
| **`HyperTFT`**  | Infinite matrix game variant of `TitForTat`. Chooses the C action on the first turn and reciprocates the opponent's last action.|
| **`Random`**        | Randomly chooses the C or D action. |
| **`TitForTat`**    | Chooses the C action on the first turn and reciprocates the opponent's last action.|


Coin Game:

|   Agent      |    Description| 
| ----------- | ----------- |
| **`EvilGreedy`** | Attempts to pick up the closest coin. If equidistant to two colored coins, then it chooses its opponents color coin.|
| **`GoodGreedy`** | Attempts to pick up the closest coin. If equidistant to two colored coins, then it chooses its own color coin. |
| **`RandomGreedy`**  | Attempts to pick up the closest coin. If equidistant to two colored coins, then it randomly chooses a color coin. |
| **`Stay`**     | Agent does not move.|

### Learning

|  Agent      |   Description | 
| ----------- | ----------- |
| **`Naive`**  | Simple learning agent that learns via REINFORCE. |
| **`NaiveEx`**  | Infinite matrix game variant of `Naive`. Simple learning agent that learns via REINFORCE. |
| **`MFOS`**  | Meta-learning algorithm for opponent shaping. |
| **`PPO`**  | Learning agent parameterised by a multilayer perceptron that learns via PPO. |
| **`PPO_memory`** | Learning agent parameterised by a multilayer perceptron with a memory component that learns via PPO. |
| **`Tabular`** | Learning agent parameterised by a single layer perceptron that learns via PPO. |



```{note}
`PPO_memory` serves as the core learning algorithm for both **Good Shepherd (GS)** and **Context and History Aware Other Shaping (CHAOS)** when the training with meta-learning.
```






