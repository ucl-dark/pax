# Agents

Pax includes a number of learning and fixed agents. They are specified in the `.yaml` files as `Agent1` and `Agent2`. Canonically, we care about the outcome of `Agent1`. 

## Interfaces

We provide an agent [interface](../pax/agents/agent.py) to help develop new agents for pax. These methods are necessary for existing runners and should be functional.

## Existing Agents
All the learning strategies have their own folder and the fixed agents can be viewed in `pax/agents/strategies.py` (inspired by Axelrods' Tournament). 

Below we list currently supported agents and their respective implementation details

| Agents | Description |
| -------| ----------- |
| `Naive`      | A learning agent that updates using REINFORCE [insert paper here]         |
| `PPO`    | A learning agent that updates using PPO [insert paper here]   |
| `PPO_memory`    | A learning agent that updates using PPO and a memory state [insert paper here]       |
| `MFOS`    | Model free opponent shaping meta-agent based upon the [ICML paper 2021](https://arxiv.org/abs/2205.01447)     |
| `GS`    | The Good Shepard meta-agent based upon the [paper](https://arxiv.org/abs/2202.10135)|
| `Defect`    | A fixed agent that always defects        |
| `Altruistic`    | A fixed agent that always cooperates        |
| `TitForTat`    | A fixed agent that cooperates on the first move and then reciprocates action of the opponent from the previous turn    |

*Note: `MFOS` and `GS` are meta-agents, so will only work with the meta environment