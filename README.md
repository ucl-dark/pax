
# Scaling Opponent Shaping to High Dimmensional Games

Code to accompy ICML submission.

# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to install JAX with the relevant accelerator support.

## General Information
The project entrypoint is `pax/experiment.py`. The simplest command to run a game would be: 

```bash
python -m pax.experiment
```

We currently use [WandB](https://wandb.ai/) for logging and [Hydra](https://hydra.cc/docs) for configs. Hyperparameters are stored `/conf/experiment` as `.yaml` files. Depending on your needs, you can specify hyperparameters through the CLI or by changing the `.yaml` files directly. 

```bash
python -m pax.experiment +total_timesteps=1_000_000 +num_envs=10
```

We currently support two major environments: `MatrixGames` and `CoinGame`.
```

For `MatrixGames`, we support the ability to specify your own payoff matrix either through the CLI or the `yaml` files. For example the common Iterated Prisoners Dilemma is:
```bash 
python -m pax.experiment +experiment/ipd=ppo ++payoff="[[-2,-2], [0,-3], [-3,0], [-1,-1]]" ++wandb.group="testing"
```

## Experiments


### Coin Game - Independent Learners
```bash 
python -m pax.experiment +experiment/cg=sanity.yaml ++wandb.group="testing" 
``` 

### Coin Game - SHAPER
```bash 
python -m pax.experiment +experiment/cg=chaos_v_ppo_mem.yaml ++wandb.group="testing" 
``` 

### Coin Game - Good Shepherd
```bash 
python -m pax.experiment +experiment/cg=gs_v_ppo_mem.yaml ++wandb.group="testing" 
``` 

### Coin Game - Model Free Opponent Shaping (ES)
```bash 
python -m pax.experiment +experiment/cg=mfos_es_v_ppo.yaml ++wandb.group="testing" 
``` 

### Coin Game - Model Free Opponent Shaping (RL)
```bash 
python -m pax.experiment +experiment/cg=mfos_rl_v_ppo.yaml ++wandb.group="testing" 
``` 