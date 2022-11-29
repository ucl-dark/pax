# Training 

Pax provides fully configurable training parameters for experiments. 

## Overview 

Training parameters allow users to fully specify the training protocol of their experiment. Users can configure the experiment `.yaml` file to specify details such as episode length, number of environments, and much more. 

```
... 
# Training
top_k: 5
popsize: 128 
num_envs: 50
num_opps: 1
num_devices: 2
num_steps: 9600
num_inner_steps: 16 
num_generations: 2000
...
```

## List of Training Parameters

### num_devices 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of devices used to train the agent. Values greater than `1` require multiple GPUs.|

```{note}
The following piece of code can used to debug multi-devices on CPU if run at the top of `experiment.py`. 
```

```
import os
from jax.config import config
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
config.update('jax_disable_jit', True)
```

### num_envs 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of environments used to train the agent.| 

### num_generations 

|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric*  | Number of generations to train the agent when training with evolution.| 

### num_inner_steps 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of inner steps within an episode. Set equal to `num_steps` when running `env: sequential`| 

### num_opps 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of opponents in each environment. Typically set to `1`.  | 

### num_steps 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of steps in a meta episode. |

Example: 
```
num_inner_steps: 16 # Episode length
num_steps: 9600     # Steps in a meta-episode
```

Following the formula `number of episodes = num_steps / num_inner_steps`, we can calculate the number of episodes. In this example, each rollout will contain 600 episodes of length 16 (`600 episodes = 9600 steps / 16 steps per episode`). 

### popsize
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric*  | Size of population when training with evolution. | 

### top_k 
|       Name | Description   | 
| :----------- | :----------- |
| *Numeric*    | Number of agents to show when training with evolution.  |   





