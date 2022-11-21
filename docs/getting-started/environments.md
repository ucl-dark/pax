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

## List of Environments

|       Name | Description   | 
| :----------- | :----------- |
|`IteratedMatrixGame`(num_inner_steps)| Classic normal form game with a 2x2 payoff matrix repeatedly played over `n` steps. |                       
|`InfiniteMatrixGame`(num_steps) | Special case of the classic normal form game that calculates an exact value, simulating an infinite game. 
|`CoinGame`(num_inner_steps, num_outer_steps, cnn, egocentric)           | Classic grid-world social dilemma environment.          |                                                 

```{note}
Docstrings are under constuction. Please check back later. 
```






