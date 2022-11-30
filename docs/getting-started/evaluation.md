# Saving & Loading

Pax provides an easy way to save and load your models. 

## Overview 

Saving and loading allows users to save or load models locally or from Weight and Biases. Users can configure the experiment `.yaml` file to set up the save and load file path, either locally or online. 

```
... 

# Save
save: True
save_interval: 10

...

# Evaluation
run_path: ucl-dark/cg/3mpgbfm2
model_path: exp/coin_game-EARL-PPO_memory-vs-Random/run-seed-0/2022-09-08_20.41.03.643377/generation_30

... 

# Logging setup
wandb:
  entity: "ucl-dark"
  project: cg
  group: 'EARL-${agent1}-vs-${agent2}'
  name: run-seed-${seed}
  log: False

```

## List of Saving and Loading Parameters

### model_path 
|       Name | Description   | 
| :----------- | :----------- |                 
|*String* | Filepath to load the model. | 

### run_path 
|       Name | Description   | 
| :----------- | :----------- |                 
|*String* | If using Weights and Biases (i.e. `wandb.log=True`), this is the  run path of the model used to load the model.  | 

### save 
|       Name | Description   | 
| :----------- | :----------- |                 
|*bool* | If `True`, the model is saved to the filepath specified by `save_dir`. |


### save_dir 
|       Name | Description   | 
| :----------- | :----------- |                 
|*String* | Filepath used to save a model. | 

### save_interval 

|       Name | Description   | 
| :----------- | :----------- |                 
|*Int*  | Number of iterations between saving a model. | 

### wandb 

```{note}
The following parameters are used for Weights and Biases specific features.  
```

```
wandb:
  entity: "ucl-dark"
  project: cg
  group: 'EARL-${agent1}-vs-${agent2}'
  name: run-seed-${seed}
  log: False
```
|       Name | Description   | 
| :----------- | :----------- |                 
|`entity` | Weights and Biases entity. |
|`project` | Weights and Biases project name.  |
|`group` | Weights and Biases group name.  |
|`name` | Weights and Biases run name.  |
|`log` | Weights and Biases run name.  |






