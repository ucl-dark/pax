# Runner 

## Overview 

Pax provides a number of experiment runners useful for different use cases of training and evaluating reinforcement learning agents. 

## Specifying a Runner

Pax centers around its runners, pieces of custom experiment logic that leverage the speed of JAX. Below is an example of that logic: 

```
def _rollout(carry, unused):
    """Runner for inner episode"""
    (
        rngs,
        obs,
        a_state,
        a_mem,
        env_state,
        env_params,
    ) = carry

    # unpack rngs
    rngs = self.split(rngs, 4)
    action, a_state, new_a_mem = agent1.batch_policy(
        a_state,
        obs[0],
        a_mem,
    )

    next_obs, env_state, rewards, done, info = env.step(
        rngs,
        env_state,
        (action, action),
        env_params,
    )

    traj = Sample(
        obs1,
        action,
        rewards[0],
        new_a1_mem.extras["log_probs"],
        new_a1_mem.extras["values"],
        done,
        a1_mem.hidden,
    )

    return (
        rngs,
        next_obs,
        a1_state,
        new_a1_mem,
        env_state,
        env_params,
    ), (
        traj1,
        traj2,
    )


agent = Agent(args)
state, memory = agent.make_initial_state(rng, init_hidden)

for _ in range(num_updates):
    final_timestep, batch_trajectory = jax.lax.scan(
        _rollout,
        ((obs, env_state, rng), rollout_length),
        10,
    )

    _, obs, rewards, a1_state, a1_mem, _, _ = final_timestep

    state, memory, stats = agent.update(
        batch_trajectory, obs[0], state, memory
    )
```

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