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

## List of Agents

```{note}
Fixed agents are game-specific, while the learning agents (i.e. PPO, Good Shepherd) can be used out-of-the-box in both games. 
```

### Fixed

Matrix games:

|        |    | 
| ----------- | ----------- |
| **`Altruistic`**(num_envs)    | Always chooses the Cooperate (C) action. |
| **`Defect`**(num_envs)       | Always chooses the Defect (D) action. |
| **`GrimTrigger`**(num_envs)   | Chooses the C action on the first turn and reciprocates with the C action until the opponent chooses D, where Grim switches to only choosing D.|
| **`Random`**(num_actions, num_envs)        | Randomly chooses the C or D action. |
| **`TitForTat`**(num_envs)    | Chooses the C action on the first turn and reciprocates the opponent's last action.|

Coin Game:

|        |    | 
| ----------- | ----------- |
| **`EvilGreedy`**(num_envs) | Attempts to pick up the closest coin. If equidistant to two colored coins, then it chooses its opponents color coin.|
| **`GoodGreedy`**(num_envs) | Attempts to pick up the closest coin. If equidistant to two colored coins, then it chooses its own color coin. |
| **`RandomGreedy`**(num_envs)  | Attempts to pick up the closest coin. If equidistant to two colored coins, then it randomly chooses a color coin. |
| **`Stay`**(num_actions, num_envs)     | Agent does not move.|

### PPO

This is a PPO agent. 

### Context and History Aware Other Shaping (CHAOS)

This is a CHAOS agent. 

### Good Shepherd (GS)

This is a GS agent. 

### Model-Free Opponent Shaping (M-FOS)

This is a M-FOS agent. 




