for SEED in 0 1 2 3 4 5 6 7 8 9
do
python3 -m pax.experiment +experiment=n_player/tft/tft_harsh_3pl_ipd.yaml ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/tft/tft_harsh_4pl_ipd.yaml ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/tft/tft_harsh_5pl_ipd.yaml ++seed=$SEED

python3 -m pax.experiment +experiment=n_player/tft/tft_soft_3pl_ipd.yaml ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/tft/tft_soft_4pl_ipd.yaml ++seed=$SEED
python3 -m pax.experiment +experiment=n_player/tft/tft_soft_5pl_ipd.yaml ++seed=$SEED
done