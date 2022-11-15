## Environments 

Pax includes many environments specified by `env_id`. These are `matrix_game` and `coin_game`. Independetly you can specify your enviroment type by `env_type`, the options supported are `sequential`, `meta` and `infinite`. These are specified in the config files in `pax/configs/{env_id}/EXPERIMENT.yaml`. 

| Environment ID | Environment Type | Description |
| ----------- | ----------- | ----------- |
|ipd| `sequential`      | An iterated matrix game with a predetermined number of timesteps per episode with a discount factor $\gamma$         |
|ipd | `infinite`    | An infinite matrix game that calculates exact returns given a payoff and discount factor $\gamma$       |
|ipd | `meta`    | A meta game over the iterated matrix game with an outer agent (player 1) and an inner agent (player 2). The inner updates every episode, while the the outer agent updates every meta-episode |
|coin_game | `sequential`    | A sequential series of episode of the coin game between two players. Each player updates at the end of an episode|
|coin_game | `meta`    | A meta learning version of the coin game with an outer agent (player 1) and an inner agent (player 2). The inner updates every episode, while the the outer agent updates every meta-episode|
