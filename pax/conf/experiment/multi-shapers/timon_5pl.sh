python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_ipd ++num_iters=2000 ++seed=1 ++pop_size=150
for SEED in  2 3 4 5
do
python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_ipd ++num_iters=2500 ++seed=$SEED ++pop_size=150
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_ipd ++num_iters=2500 ++seed=$SEED ++pop_size=150
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_ipd ++num_iters=2000 ++seed=$SEED ++pop_size=150
done