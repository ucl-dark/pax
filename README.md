
# Scaling Opponent Shaping to High Dimmensional Games

Code to accompy ICML submission 2581.

# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to install JAX with the relevant accelerator support.

## General Information
The project entrypoint is `pax/experiment.py`. The simplest command to run a game would be: 

```bash
python -m pax.experiment
```

We currently use [WandB](https://wandb.ai/) for logging and [Hydra](https://hydra.cc/docs) for configs. Hyperparameters are stored `/conf/experiment` as `.yaml` files. Depending on your needs, you can specify hyperparameters through the CLI. WandB is required to make plots.

## Experiments

### Iterated Prisoners Dilemma - Independent Learners
```bash 
python -m pax.experiment +experiment/ipd=sanity.yaml
``` 

### Coin Game - Independent Learners
```bash 
python -m pax.experiment +experiment/cg=sanity.yaml
``` 

### Coin Game - SHAPER
```bash 
python -m pax.experiment +experiment/cg=chaos_v_ppo_mem.yaml
``` 

### Coin Game - Good Shepherd
```bash 
python -m pax.experiment +experiment/cg=gs_v_ppo_mem.yaml
``` 

### Coin Game - Model Free Opponent Shaping (ES)
```bash 
python -m pax.experiment +experiment/cg=mfos_es_v_ppo.yaml 
``` 

### Coin Game - Model Free Opponent Shaping (RL)
```bash 
python -m pax.experiment +experiment/cg=mfos_rl_v_ppo.yaml
``` 

### IPD in the Matrix - Independent Learners
```bash 
python -m pax.experiment +experiment/ipditm=sanity.yaml
``` 

### IPD in the Matrix - SHAPER
```bash 
python -m pax.experiment +experiment/ipditm=chaos_v_ppo_mem.yaml
``` 

### IMP in the Matrix - Good Shepherd
```bash 
python -m pax.experiment +experiment/ipditm=gs_v_ppo_mem.yaml 
``` 

### IPD in the Matrix - Model Free Opponent Shaping (ES)
```bash 
python -m pax.experiment +experiment/ipditm=mfos_es_v_ppo.yaml
``` 

### IPD in the Matrix  - Model Free Opponent Shaping (RL)
```bash 
python -m pax.experiment +experiment/ipditm=mfos_rl_v_ppo.yaml
``` 