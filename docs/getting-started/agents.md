# Agents 

We provide a number of fixed opponents and learning agents to train and train against. 

## Fixed

For matrix games: 

|        |    | 
| ----------- | ----------- |
| **`Altruistic`**    | Always chooses the Cooperate (C) action. |
| **`Defect`**        | Always chooses the Defect (D) action. |
| **`GrimTrigger`**   | Chooses the C action on the first turn and reciprocates with the C action until the opponent chooses D, where Grim switches to only choosing D.|
| **`Random`**        | Randomly chooses the C or D action. |
| **`TitForTat`**     | Chooses the C action on the first turn and reciprocates the opponent's last action.|

For Coin Game: 

|        |    | 
| ----------- | ----------- |
| **`EvilGreedy`** | Attempts to pick up the closest coin. If equidistant to two colored coins, then it chooses its opponents color coin.|
| **`GoodGreedy`** | Attempts to pick up the closest coin. If equidistant to two colored coins, then it chooses its own color coin. |
| **`RandomGreedy`**  | Attempts to pick up the closest coin. If equidistant to two colored coins, then it randomly chooses a color coin. |
| **`Stay`**     | Agent does not move.|

## PPO

This is a PPO agent. 

## Context and History Aware Other Shaping (CHAOS)

This is a CHAOS agent. 

## Good Shepherd (GS)

This is a GS agent. 

## Model-Free Opponent Shaping (M-FOS)

This is a M-FOS agent. 




