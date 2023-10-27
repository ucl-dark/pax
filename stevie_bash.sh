#!/bin/bash
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/two=mfos_avg_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/two=mfos_avg_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/two=mfos_avg_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/two=mfos_avg_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/two=mfos_avg_4 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/ten=mfos_avg_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/ten=mfos_avg_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/ten=mfos_avg_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/ten=mfos_avg_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/ten=mfos_avg_4 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/twenty=mfos_avg_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/twenty=mfos_avg_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/twenty=mfos_avg_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/twenty=mfos_avg_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_avg/twenty=mfos_avg_4 ++wandb.log=True


###### MFOS NOTHING ######
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/two=mfos_nothing_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/two=mfos_nothing_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/two=mfos_nothing_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/two=mfos_nothing_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/two=mfos_nothing_4 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/ten=mfos_nothing_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/ten=mfos_nothing_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/ten=mfos_nothing_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/ten=mfos_nothing_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/ten=mfos_nothing_4 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/twenty=mfos_nothing_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/twenty=mfos_nothing_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/twenty=mfos_nothing_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/twenty=mfos_nothing_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/mfos_nothing/twenty=mfos_nothing_4 ++wandb.log=True

###### SHAPER AVG ######
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/two=shaper_avg_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/two=shaper_avg_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/two=shaper_avg_2 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/ten=shaper_avg_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/ten=shaper_avg_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/ten=shaper_avg_2 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/twenty=shaper_avg_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/twenty=shaper_avg_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_avg/twenty=shaper_avg_2 ++wandb.log=True

###### SHAPER NOTHING ######
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/two=shaper_nothing_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/two=shaper_nothing_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/two=shaper_nothing_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/two=shaper_nothing_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/two=shaper_nothing_4 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/ten=shaper_nothing_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/ten=shaper_nothing_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/ten=shaper_nothing_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/ten=shaper_nothing_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/ten=shaper_nothing_4 ++wandb.log=True

python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/twenty=shaper_nothing_0 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/twenty=shaper_nothing_1 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/twenty=shaper_nothing_2 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/twenty=shaper_nothing_3 ++wandb.log=True
python -m pax.experiment -m +experiment/ipd/stevie/shaper_nothing/twenty=shaper_nothing_4 ++wandb.log=True