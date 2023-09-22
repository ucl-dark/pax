
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_ipd  ++seed=$SEED ++wandb.group='op1_naive_3pl_ipd' ++num_opps=1  ++num_envs=100  ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_sd  ++seed=$SEED ++wandb.group='op1_naive_3pl_sd' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_sh ++seed=$SEED ++wandb.group='op1_naive_3pl_sh' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_tc ++seed=$SEED ++wandb.group='op1_naive_3pl_tc' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_ipd ++seed=$SEED ++wandb.group='op1_naive_4pl_ipd' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_sd ++seed=$SEED ++wandb.group='op1_naive_4pl_sd' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_sh ++seed=$SEED ++wandb.group='op1_naive_4pl_sh' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_tc ++seed=$SEED ++wandb.group='op1_naive_4pl_tc' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_ipd ++seed=$SEED ++wandb.group='op1_naive_5pl_ipd' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_sd ++seed=$SEED ++wandb.group='op1_naive_5pl_sd' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_sh ++seed=$SEED ++wandb.group='op1_naive_5pl_sh' ++num_opps=1 ++num_envs=100 ++num_iters=3000
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_tc ++seed=$SEED ++wandb.group='op1_naive_5pl_tc' ++num_opps=1 ++num_envs=100 ++num_iters=3000
done