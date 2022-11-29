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
|*Numeric* | Number of devices used to train the agent. Requires access to multiple GPUs.|

### num_envs 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of environments to train the agent.| 

### num_generations 

|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric*  | Number of generations to train the agent when training with evolution.| 

### num_inner_steps 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of inner steps within an episode. | 

### num_opps 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of opponents in each environment. | 

### num_steps 
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric* | Number of outer steps in a meta episode. | 

### popsize
|       Name | Description   | 
| :----------- | :----------- |                 
|*Numeric*  | Size of population when training with evolution. | 

### top_k 
|       Name | Description   | 
| :----------- | :----------- |
| *Numeric*    | Number of agents to show when training with evolution.  |   





