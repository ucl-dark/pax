# quick ones
for SEED in 0 1 2 3 4 
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_sd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_sd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_sh ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_tc ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_sd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_sh ++seed=$SEED

done


# slow ones
for SEED in 0 1 2 3 4 
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_ipd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/3pl_tc ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_ipd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_tc ++seed=$SEED
done



# not converging ones
for SEED in 0 1 2 3 4 
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_sh ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_ipd ++seed=$SEED
done