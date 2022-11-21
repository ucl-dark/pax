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

## Options

|       Name | Description   | 
| :----------- | :----------- |
|`IteratedMatrixGame`| Classic normal form game with a 2x2 payoff matrix repeatedly played over `n` steps. |                       
|`InfiniteMatrixGame` | Special case of the classic normal form game that calculates an exact value, simulating an infinite game. 
|`CoinGame`           | Classic grid-world social dilemma environment.          |                                                 

<!-- ## Matrix Games
We include iterated and infinite matrix games. 

$$ 
\begin{bmatrix}
\text{CC} & \text{DC}  \\
\text{CD} & \text{DD}  \\
\end{bmatrix} 
$$  -->

<!-- ## Coin Game

We include the classic 3 x 3 grid-style Coin Game.  -->







