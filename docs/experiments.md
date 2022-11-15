Here is a list of existing Parity tests we have.

Canonically, the `Agent 1` agent type is constant in each experiment while `Agent 2` changes. 

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