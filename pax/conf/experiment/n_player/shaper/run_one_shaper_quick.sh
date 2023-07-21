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