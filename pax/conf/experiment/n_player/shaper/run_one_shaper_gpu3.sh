# slow ones but they do converge
for SEED in 1 2 3 4 5
do
python3 -m pax.experiment +experiment=n_player/shaper/3pl_sh ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/4pl_ipd ++seed=$SEED
done