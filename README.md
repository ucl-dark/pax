# Pax
Here we start building Pax - a Jax Batched Environment for sequential matrix games.

Pax has a simple interface, similar to [dm env](https://github.com/deepmind/dm_env) and aspires to work with agents from both [acme](https://github.com/deepmind/acme) and [magi](https://github.com/ethanluoyc/magi).

```python
import IteratedPrisonersDilemma
import jax.numpy as jnp 

env = IteratedPrisonersDilemma(episode_length, num_envs)
t0, t1 = env.reset()
while not (t0.last() and t1.last()):
     a0: jnp.array = agent.step(timestep_0.observation)
     a1: jnp.array = agent.step(timestep_1.observation)
     t0, t1 = env.step(a0, a1)
```

and timestep returns the following:

```python
timestep.observation.shape()
# (num_envs, num_states)
timestep.reward.shape()
# (num_envs, 1)
```

## Current State

- [x] PrisonersDilemmas Environment
- [x] Deterministic tit-for-tat player
- [x] Re-structure to use state space
- [x] Add method to vanilla train SAC
- [x] Add Observation Function
- [x] Change state integers into one hot encodings
- [x] Need to change individual actions into state
- [X] SAC to run using categorical distribution
- [X] Make agent experiment against another agent
- [X] Start saving models post train runs
- [X] Make a specific different state for first play (no history)
- [ ] Add memory to agents (RNN based policies)
- [ ] Policy network
- [ ] Behaviour Cloning script
- [ ] Standard PPO script


Nice to haves:
- [X] Batched Game Environment (with tests)
- [ ] Make pay-off matrix configurable
- [x] Make environment jit-able
- [ ] Clean up requirements
