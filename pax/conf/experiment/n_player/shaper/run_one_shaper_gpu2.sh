# slow ones but they do converge
for SEED in 2 3 4 5
do
python3 -m pax.experiment +experiment=n_player/shaper/5pl_ipd ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/shaper/5pl_tc ++seed=$SEED
done