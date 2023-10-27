# Runners 

## Evo Runner

The Evo Runner optimizes the first agent using evolutionary learning. 

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/cg/mfos.yaml) for an example of how to configure it.

## Evo Runner N-Roles

This runner extends the evo runner to `N > 2` agents by letting the first and second agent assume multiple roles that can be configured via `agent1_roles` and `agent2_roles` in the experiment configuration.
Both agents receive different sets of memories for each role that they assume but share the weights.

- For heterogeneous games roles can be shuffled for each rollout using the `shuffle_players` flag. 
- Using the `self_play_anneal` flag one can anneal the self-play probability from 0 to 1 over the course of the experiment.

See [this experiment](https://github.com/akbir/pax/blob/bb0e69ef71fd01ec9c85753814ffba3c5cb77935/pax/conf/experiment/rice/shaper_v_ppo.yaml) for an example of how to configure it.

## Weight sharing Runner

A simple baseline for MARL experiments is having one agent assume multiple roles and share the weights between them (but not the memory).
In order for this approach to work the observation vector needs to include one entry that indicates the role of the agent (see [Terry et al.](https://arxiv.org/abs/2005.13625v7).

See [this experiment](https://github.com/akbir/pax/blob/9d3fa62e34279a338c07cffcbf208edc8a95e7ba/pax/conf/experiment/rice/weight_sharing.yaml) for an example of how to configure it.

## Evo Hardstop

The Evo Runner optimizes the first agent using evolutionary learning. 
This runner stops the learning of an opponent during training, corresponds to the hardstop challenge of Shaper.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Scanned

The Evo Runner optimizes the first agent using evolutionary learning. 
Here we also scan over the evolutionary steps, which makes compilation longer, training shorter and logging stats is not possible.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Mixed LR Runner (experimental)

The Evo Runner optimizes the first agent using evolutionary learning. 
This runner randomly samples learning rates for the opponents.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Mixed Payoff (experimental)

The Evo Runner optimizes the first agent using evolutionary learning. 
Payoff matrix is randomly sampled at each rollout. Each opponent has a different payoff matrix.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Mixed Payoff Gen (experimental)

The Evo Runner optimizes the first agent using evolutionary learning. 
Payoff matrix is randomly sampled at each rollout. Each opponent has the same payoff matrix.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Mixed IPD Payoff (experimental)

The Evo Runner optimizes the first agent using evolutionary learning. 
This runner randomly samples payoffs that follow Iterated Prisoner's Dilemma [constraints](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma).

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Mixed Payoff Input (experimental)

The Evo Runner optimizes the first agent using evolutionary learning. 
Payoff matrix is randomly sampled at each rollout. Each opponent has the same payoff matrix. The payoff matrix is observed as input to the agent.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.

## Evo Mixed Payoff Only Opp (experimental)

The Evo Runner optimizes the first agent using evolutionary learning. 
Noise is added to the opponents IPD-like payout matrix at each rollout. Each opponent has the same noise added.

See [this experiment](https://github.com/akbir/pax/blob/9a01bae33dcb2f812977be388751393f570957e9/pax/conf/experiment/ipd/shaper_att_v_tabular.yaml) for an example of how to configure it.



