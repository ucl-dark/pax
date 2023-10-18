# Notes on runners

## MARL runner

Main agent semantics of the training loop:

```psuedo
init env and agents

_rollout for each iteration:
    agent1.reset_memory()
    if env meta: agent2.init()
    env.reset()
    
    _outer_rollout for each outer_step:
        
        _inner_rollout for each inner_step:
            agent1.act()
            agent2.act()
            env.step()
        
    
    agent1.reset_memory()
    agent2.reset_memory()
            
```
