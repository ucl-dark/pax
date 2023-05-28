
for SEED in 0 1 2 3 4 5 6 7 8 9
do
    python3 -m pax.experiment +experiment=n_player/indep/naive_2pl_sd ++seed=$SEED
    python3 -m pax.experiment +experiment=n_player/indep/naive_2pl_sh ++seed=$SEED 

    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_ipd  ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_sd  ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_sh ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_3pl_tc ++seed=$SEED 

    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_ipd ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_sd ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_sh ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_4pl_tc ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_ipd ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_sd ++seed=$SEED
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_sh ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/indep/naive_5pl_tc ++seed=$SEED 
done