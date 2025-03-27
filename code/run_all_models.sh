# Run all models for analysis
conda activate movement_clean

## NOTE: Don't try to run all of these simultaneously.  It will take a long time.
## This is just to show the sequence 

# Baseline model
python3 model0-2_simulation.py model_parameters.yml
python3 model3_simulation.py model_parameters.yml

# Wrap landscape
python3 model0-2_simulation.py model_parameters_wrap_landscape.yml
python3 model3_simulation.py model_parameters_wrap_landscape.yml

# More dispersal
python3 model0-2_simulation.py model_parameters_all_yearlings_disperse.yml
python3 model3_simulation.py model_parameters_all_yearlings_disperse.yml

# High density
python3 model0-2_simulation.py model_parameters_high_density.yml
python3 model3_simulation.py model_parameters_high_density.yml

# External infection
python3 model3_simulation.py model_parameters_external_infection.yml
