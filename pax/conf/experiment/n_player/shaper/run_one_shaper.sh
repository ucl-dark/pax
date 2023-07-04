# quick ones to converge
for SEED in 1 2 3 4 5
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_sd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_sd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_sh ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_tc ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_sd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_sh ++seed=$SEED
done

# slow ones but they do converge
for SEED in 1 2 3 4 5
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_ipd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/3pl_tc ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_ipd ++seed=$SEED
done

# I haven't ran this one before - I think it will converge reaosnable but check how fast
for SEED in 1 2 3 4 5 
do
python3 -m pax.experiment +experiment=n_player/shaper/5pl_tc ++seed=$SEED
done

# not converging ones pls help :(
for SEED in 1 2 3 4 5
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_sh ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_ipd ++seed=$SEED
done