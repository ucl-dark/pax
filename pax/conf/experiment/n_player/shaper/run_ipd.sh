for SEED in 0 1 2 3 4 
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_ipd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_ipd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_ipd ++seed=$SEED
done