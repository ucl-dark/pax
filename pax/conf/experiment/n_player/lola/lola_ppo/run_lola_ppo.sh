for SEED in 0 1 2 
do
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_3pl_ipd  ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_3pl_sd  ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_3pl_sh ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_3pl_tc ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_4pl_ipd ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_4pl_sd ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_4pl_sh ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_4pl_tc ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_5pl_ipd ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_5pl_sd ++seed=$SEED
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_5pl_sh ++seed=$SEED 
    python3 -m pax.experiment +experiment=n_player/lola/lola_ppo/lola_5pl_tc ++seed=$SEED 
done