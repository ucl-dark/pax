for SEED in  1 2 3 4 5
do
#sh
python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_sh ++num_iters=100 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_sh ++num_iters=100 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_sh ++num_iters=150 ++seed=$SEED

#sd
python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_sd ++num_iters=150 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_sd ++num_iters=150 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_sd ++num_iters=150 ++seed=$SEED

#tc
python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_tc ++num_iters=150 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_tc ++num_iters=500 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_tc ++num_iters=200 ++seed=$SEED

#ipd - pls check max popsize that fits, these take ages to converge
python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_ipd ++num_iters=2500 ++seed=$SEED ++pop_size=150
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_ipd ++num_iters=2500 ++seed=$SEED ++pop_size=150
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_ipd ++num_iters=2000 ++seed=$SEED ++pop_size=150
done