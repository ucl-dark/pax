# Environments

## Overview 
Pax supports two environments for learning agents to train within: matrix games and grid-world games. 

## Specifying the Environment

Pax environments are similar to gymnax. To specify an environment, import the environment and specify the environment parameters. 

```
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

To specify the parameters for the environment: 

```
...
# Environment  
env_id: coin_game
env_type: meta
egocentric: True
env_discount: 0.96
payoff: [[1, 1, -2], [1, 1, -2]]
...
```

## List of Environment Parameters

### env_id 
|       Name | Description   | 
| :----------- | :----------- |
|`iterated_matrix_game`| Classic normal form game with a 2x2 payoff matrix repeatedly played over `n` steps. |                       
|`infinite_matrix_game` | Special case of the classic normal form game that calculates an exact value, simulating an infinite game. 
|`coin_game`    | Classic grid-world social dilemma environment.          |               

### env_type

|       Name | Description   | 
| :----------- | :----------- |
|`sequential`| Classic normal form game with a 2x2 payoff matrix repeatedly played over `n` steps. |                       
|`meta`| Meta-learning regime, where an agent learns via meta-learning.     |

### egocentric 
|       Name | Description   | 
| :----------- | :----------- |
|*bool*| If `True`, sets an agent in the Coin Game environment to an egocentric view, empirically found to be more appropriate for other shaping. Else, sets an agent in  to a non-egocentric view, in line with the original version. |

### env_discount 
<!-- TODO: Possibly deprecate. -->
|       Name | Description   | 
| :----------- | :----------- |
|*Numeric*| Meta-learning discount factor. Between 0 and 1. |     

### payoff 
|       Name | Description   | 
| :----------- | :----------- |
|*Array*| Custom payoff for game. |                       

Example: 

```
# if playing Coin Game 
payoff: [[1, 1, -2], [1, 1, -2]]
```

```
# if playing Matrix Games
payoff: [[-1, -1], [-3, 0], [0, -3], [-2, -2]]
```

```{note}
Docstrings are under constuction. Please check back later. 
```





