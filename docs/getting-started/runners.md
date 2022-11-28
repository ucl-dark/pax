# Runner 

## Overview 

Pax provides a number of experiment runners useful for different use cases of training and evaluating reinforcement learning agents. 

## Specifying a Runner

To specify the runner in an experiment, use a pre-made `.yaml` file located in `conf/...` or create your own, and specify the runner with `runner`. In the below example, the `evo` flag and the `EvoRunner` used.

```
...
# Runner 
runner: evo 
...
```

## Runners 
|   Runner      |    Description| 
| ----------- | ----------- |
| **`eval`**   | Evaluation runner, where a single, pre-trained agent is evaluated. |
| **`evo`** | Evolution runner, where two independent agents are trained via Evolutionary Strategies (ES). |
| **`rl`** | Multi-agent runner, where two independent agents are trained via reinforcement learning.  |
| **`sarl`**  | Single-agent runner, where a single agent is trained via reinforcement learning.  |