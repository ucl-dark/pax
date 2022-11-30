# Training 

Pax provides fully configurable training parameters for experiments. 

## Overview 

Training parameters allow users to fully specify the training protocol of their experiment. Users can configure the experiment `.yaml` file to specify details such as episode length, number of environments, and much more. 

## List of Training Parameters

### ppo
<!-- 
TODO: 
- with_memory is possibly deprecated.
- with_cnn is possibly deprecated. 
- kernel_shape is possibly deprecated. 
- separate is possibly deprecated. 
- possibly make this like the other .md files without a table. 
  -->

|       Name | Type | Description   | 
| :----------- | :----------- | :----------- |
| `num_minibatches`   | *int*| Number of minibatches.  |   
| `num_epochs`   | *int* | Number of epochs.   |   
| `gamma`   | *Numeric*| Discount factor $\gamma$.  |   
| `gae_lambda`   | *Numeric*| Generalized advantage estimate $\lambda$ factor.  |   
| `ppo_clipping_epsilon`   | *Numeric*| Clipping factor $\epsilon$.   |   
| `value_coeff`   | *Numeric*| Value coefficient.   |   
| `clip_value`   | *Numeric*| Clip value.   |   
| `max_gradient_norm`   | *Numeric*| Max gradient norm.   |   
| `anneal_entropy`   | *bool* | Whether to anneal the entropy term.   |   
| `entropy_coeff_start`   | *Numeric*| Starting entropy annealing coefficient.  |   
| `entropy_coeff_horizon`   |*Numeric*|  Number of iterations before entropy coefficient reaches `entropy_coeff_end`   |   
| `entropy_coeff_end`   | *Numeric*| Ending entropy annealing coefficient.  |   
| `lr_scheduling`   | *bool* | Whether to annealing the learning rate.   |   
| `learning_rate`   | *Numeric*| Learning rate.   |   
| `adam_epsilon`   | *Numeric*| Adam epsilon.   |   
| `with_memory`   | *bool*| Whether to use memory.  |   
| `with_cnn`   |*bool* | Whether to use a CNN in Coin Game.  |   
| `output_channels`   | *int*| Number of output channels.   |   
| `kernel_shape`   | *Array*|  Size of kernel shape.   |   
| `separate`   | *bool*| Whether to use separate networks in CNN.  |   
| `hidden_size`   | *Numeric*| Hidden size of memory layer.  |   

Example
```
# config.yaml
ppo:
  num_minibatches: 8
  num_epochs: 2 
  gamma: 0.96
  gae_lambda: 0.95
  ppo_clipping_epsilon: 0.2
  value_coeff: 0.5
  clip_value: True
  ...  
```

### es 

|       Name | Type | Description   | 
| :----------- | :----------- | :----------- |
| `algo`   | *String*| Algorithm to use. Currently supports `[OpenES, CMA_ES, SimpleGA]`|   
| `sigma_init`   | *Numeric* | Initial scale of isotropic Gaussian noise   |   
| `sigma_decay`   | *Numeric*| Multiplicative decay factor  |   
| `sigma_limit`   | *Numeric*| Smallest possible scale  |   
| `init_min`   | *Numeric*| Range of parameter mean initialization - Min  |   
| `init_max`   | *Numeric*| Range of parameter mean initialization - Max  |   
| `clip_min`   | *Numeric*| Range of parameter proposals - Min  |   
| `clip_max`   | *Numeric*| Range of parameter proposals - Max  |   
| `lrate_init`   | *Numeric* | Initial learning rate   |   
| `lrate_decay`   | *Numeric*| Multiplicative decay factor |   
| `lrate_limit`   |*Numeric*|  Smallest possible lrate |   
| `beta_1`   | *Numeric*| Adam - beta_1 |   
| `beta_2`   | *Numeric* | Adam - beta_2 |   
| `eps`   | *Numeric*| eps constant,  |   
| `elite_ratio`   | *Numeric*| Percentage of elites to keep.  |   

Example
```
# config.yaml 
es: 
  algo: OpenES        
  sigma_init: 0.04   
  sigma_decay: 0.999  
  sigma_limit: 0.01  
  init_min: 0.0       
  init_max: 0.0       
  clip_min: -1e10     
  clip_max: 1e10     
  lrate_init: 0.1    
  lrate_decay: 0.9999 
  lrate_limit: 0.001  
  beta_1: 0.99        
  beta_2: 0.999       
  eps: 1e-8           
  elite_ratio: 0.1
```

## List of Training Hyperparameters

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

Example
```
# config.yaml
top_k: 5
popsize: 128 
num_envs: 50
num_opps: 1
num_devices: 2
num_steps: 9600
num_inner_steps: 16 
num_generations: 2000
```