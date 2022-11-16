## Environments 

Pax includes many environments specified by `env_id`. These are `infinite_matrix_game`, `iterated_matrix_game` and `coin_game`. Independetly you can specify your enviroment type as either a meta environment (with an inner/ outer loop) by `env_type`, the options supported are `sequential` or `meta`.

These are specified in the config files in `pax/configs/{env_id}/EXPERIMENT.yaml`. 

| Environment ID | Environment Type | Description |
| ----------- | ----------- | ----------- |
|`iterated_matrix_game`| `sequential`      | An iterated matrix game with a predetermined number of timesteps per episode with a discount factor $\gamma$         |
|`iterated_matrix_game` | `meta`    | A meta game over the iterated matrix game with an outer agent (player 1) and an inner agent (player 2). The inner updates every episode, while the the outer agent updates every meta-episode |
|`infinite_matrix_game` | `meta`| An infinite matrix game that calculates exact returns given a payoff and discount factor $\gamma$       |
|coin_game | `sequential`    | A sequential series of episode of the coin game between two players. Each player updates at the end of an episode|
|coin_game | `meta`    | A meta learning version of the coin game with an outer agent (player 1) and an inner agent (player 2). The inner updates every episode, while the the outer agent updates every meta-episode|
