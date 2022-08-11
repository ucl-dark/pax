# Pax
Here we start building Pax - a Jax Batched Environment for sequential matrix games (such as social dilemmas).

> *Pax (noun) - a period of peace that has been forced on a large area, such as an empire or even the whole world*

PAX has a simple interface, similar to [dm-env](https://github.com/deepmind/dm_env) and aspires to work with agents from both [acme](https://github.com/deepmind/acme) and [magi](https://github.com/ethanluoyc/magi).

```python
import IteratedPrisonersDilemma

env = IteratedPrisonersDilemma(episode_length, num_envs)
timesteps = env.reset()

agents = IndependentLearners(
     agent_0,
     agent_1
)

while not timestep[0].last():
     actions = agents.step(timesteps)
     timestep = env.step(actions))
```

and timestep returns the following:

```python
timestep = timesteps[0]
timestep.observation.shape()
# (num_envs, num_states)
timestep.reward.shape()
# (num_envs, )
```

# Installation
Pax is written in pure Python, but depends on C++ code via JAX.

Because JAX installation is different depending on your CUDA version, Haiku does not list JAX as a dependency in requirements.txt.

First, follow these instructions to install JAX with the relevant accelerator support.

We include four default games: [Iterated Prisoner's Dilemma](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma), [Stag Hunt](https://en.wikipedia.org/wiki/Stag_hunt), [Battle of the Sexes](https://en.wikipedia.org/wiki/Battle_of_the_sexes_(game_theory)), and [Chicken](https://en.wikipedia.org/wiki/Chicken_(game)). They can be called using the following flags: ```ipd, stag, sexes, chicken```. For example: 

```bash 
python -m pax.experiment game="ipd" wandb.group="testing"
``` 

The payoff matrices are as follows: 
```     
#             CC      DC     CD     DD
# IPD       = [[2,2], [3,0], [0,3], [1,1]]
# Stag hunt = [[4,4], [3,1], [1,3], [2,2]]
# BotS      = [[3,2], [0,0], [0,0], [2,3]]
# Chicken   = [[0,0], [1,-1],[-1,1],[-2,-2]]
``` 

Additionally, we support the ability to specify your own payoff matrix: 

```bash 
python -m pax.experiment payoff="[[2,2], [3,0], [0,3], [1,1]]"  wandb.group="testing"
```

# Experiments
We present an example experimentation script at `pax/experiment.py`. We currently use `wandb` for logging.

```bash 
python -m pax.experiment wandb.group="testing"
``` 

We store previous experiments as parity tests. We use [Hydra](https://hydra.cc/docs) to store these configs and keep track of good hyper-paremters. As a rule for development, we try retain backwards compatability and allow all previous results to be replicated. These can be run easily by `python -m pax.experiment +experiment=NAME` Below are a list of our existing experiments and expected results:

- `dqn` - Agent1 is a Deep Q Learning Network playing against TitForTat agent in a single sequential matrix game. Both payoffs and second player can be changed.
- 'marl2_fixed` - Agent1 is a PPO agent with memory playing a series of sequential matrix games. Second player is considered fixed but can be changed for any other agent.
- 'marl2_nl`- Agent1 is a PPO agent with memory playing a series of sequential matrix games. Second player is a naive learner who updates at the end of each sequential matrix game.
- `mfos_fixed` - Agent1 is a PPO agent which plays a series of infinite matrix games. Second player is assumed to be fixed but can changed for any other agent.
- `mfos_nl` - Agent1 is a PPO agent which plays a series of infinite matrix games. Second player is a naive learner trained via gradient ascent.
- `naive` - Agent1 is a naive Reinforcement Learner, playing a single sequential matrix game.
- `PPO` - Agent1 is a PPO agent, playing a single sequential matrix game.
- `naive` - Agent1 is a PPO agent with memory, playing a single sequential matrix game.


