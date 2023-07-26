for SEED in  1 2
do
# sh 
python3 -m pax.experiment +experiment=multi-shapers/n3pl_2shap_sh  ++num_outer_steps=1000  ++num_iters=500 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n4pl_2shap_sh  ++num_outer_steps=1000  ++num_iters=200 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n4pl_3shap_sh  ++num_outer_steps=1000  ++num_iters=200 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_sh  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_sh  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_sh  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED

# sd
python3 -m pax.experiment +experiment=multi-shapers/n3pl_2shap_sd  ++num_outer_steps=1000  ++num_iters=200 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n4pl_2shap_sd  ++num_outer_steps=1000  ++num_iters=200 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n4pl_3shap_sd  ++num_outer_steps=1000  ++num_iters=200 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_sd  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_sd  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_sd  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED

# tc
python3 -m pax.experiment +experiment=multi-shapers/n3pl_2shap_tc  ++num_outer_steps=1000  ++num_iters=500 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n4pl_2shap_tc  ++num_outer_steps=1000  ++num_iters=150 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n4pl_3shap_tc  ++num_outer_steps=1000  ++num_iters=150 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_tc  ++num_outer_steps=1000  ++num_iters=150 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_tc  ++num_outer_steps=1000  ++num_iters=500 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_tc  ++num_outer_steps=1000  ++num_iters=300 ++seed=$SEED

# ipd
python3 -m pax.experiment +experiment=multi-shapers/n3pl_2shap_ipd  ++num_outer_steps=1000  ++num_iters=2000 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n4pl_2shap_ipd ++num_outer_steps=1000  ++num_iters=1500 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n4pl_3shap_ipd ++num_outer_steps=1000  ++num_iters=1500 ++seed=$SEED

python3 -m pax.experiment +experiment=multi-shapers/n5pl_2shap_ipd  ++num_outer_steps=1000  ++num_iters=1500 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_3shap_ipd  ++num_outer_steps=1000  ++num_iters=1500 ++seed=$SEED
python3 -m pax.experiment +experiment=multi-shapers/n5pl_4shap_ipd  ++num_outer_steps=1000  ++num_iters=2000 ++seed=$SEED



done