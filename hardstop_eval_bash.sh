#!/bin/bash
###### MFOS AVG ######
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/4ykf9oe8 ++model_path=exp/MFOS-vs-Tabular/run-seed-23-pop-size-1000/2023-05-11_14.58.45.927266/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1 
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/eopf93re ++model_path=exp/MFOS-vs-Tabular/run-seed-65-pop-size-1000/2023-05-11_20.31.48.530245/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/1sqbd09n ++model_path=exp/MFOS-vs-Tabular/run-seed-47-pop-size-1000/2023-05-11_17.45.03.318240/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/3n7l8ods ++model_path=exp/MFOS-vs-Tabular/run-seed-8-pop-size-1000/2023-05-11_12.12.19.914211/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/4mf1ecxq ++model_path=exp/MFOS-vs-Tabular/run-seed-6-pop-size-1000/2023-05-11_09.25.40.656392/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1

python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/4ykf9oe8 ++model_path=exp/MFOS-vs-Tabular/run-seed-23-pop-size-1000/2023-05-11_14.58.45.927266/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/eopf93re ++model_path=exp/MFOS-vs-Tabular/run-seed-65-pop-size-1000/2023-05-11_20.31.48.530245/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/1sqbd09n ++model_path=exp/MFOS-vs-Tabular/run-seed-47-pop-size-1000/2023-05-11_17.45.03.318240/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/3n7l8ods ++model_path=exp/MFOS-vs-Tabular/run-seed-8-pop-size-1000/2023-05-11_12.12.19.914211/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=mfos_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/4mf1ecxq ++model_path=exp/MFOS-vs-Tabular/run-seed-6-pop-size-1000/2023-05-11_09.25.40.656392/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100

###### Shaper Nothing #$$$
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/2m3wh5g7 ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-65-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_22.17.07.592872/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/1jk5zly5 ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-47-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_20.46.58.588813/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/1cvpiolk ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-23-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_19.16.56.990716/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/3vml0wjy ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-6-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_16.16.19.180942/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=100

python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/2m3wh5g7 ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-65-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_22.17.07.592872/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/1jk5zly5 ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-47-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_20.46.58.588813/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/1cvpiolk ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-23-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_19.16.56.990716/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1
python -m pax.experiment -m +experiment/ipd=shaper_att_v_tabular_hardstop_eval ++wandb.log=True ++run_path=ucl-dark/ipd/3vml0wjy ++model_path=exp/EARL-Shaper-vs-Tabular/run-seed-6-OpenES-pop-size-1000-num-opps-10-att-type-nothing/2023-05-14_16.16.19.180942/generation_900 ++seed=85768,785678,764578,89678,97869,4567456,856778,3456347,45673,83346 ++stop=1