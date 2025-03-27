The repository `scv2_persistence_for_github` contains the code to reproduce the analyses described in the manuscript **A white-tailed deer population is unlikely a reservoir for SARS-CoV-2, despite multiple exposures**

Here are the steps to reproduce the results

1. Use conda to activate the conda environment from the provided `environment.yml` file (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).  The environment is called `movement_clean`.
2. Run the script `run_all_models.sh`
3. Interactively generate plots using `plot_simulations.ipynb`

Note that due additional ongoing research that used the movement data used in this manuscript, the full movement dataset is not shared. Thus, note all of the code provided can be run as it requires the movement data.

The directory contains the following folders and files

- `environment.yml`:  The conda environment file that can be used to generate the Python environment used to perform all of the analyses.  You need to build and activate this environment before running any of the scripts.
- `code`
	- `generate_UDs.py`: Generates the seasonal UDs for all deer.
	- `generate_correlation_surfaces.py`: Computes CSR ratios for all deer-by-season pairs and makes Appendix S2: Figure S1.
	- `landscape_simulation.py`:  Functions and objects used in the modeling analysis.  See documentation.
	- `model0-2_simulation.py`:  Script to build and simulate epidemiological landscapes for Model 1 - Model 3, as described in the main text. (Note that in the code we refer to these models and models 0 - 2)
	- `model3_simulation.py`: Script to build and simulate epidemiological landscapes for Model 4. (note that in the code we refer to this model as model 3)
	- `model_parameters.yml`:  Baseline model parameters for the main text simulations.
	- `model_parameters_all_yearlings_disperse.yml`:  Model parameters when all yearlings disperse
	- `model_parameters_external_infection.yml`: Model parameters when there is external infection 
	- `model_parameters_high_density.yml`: Model parameters when deer density is high on the landscape
	- `model_parameters_wrap_landscape.yml`: Model parameters when the landscape is wrapped.
	- `plot_simulations.ipynb`: Jupyter notebooks that plots the simulation results and generates the plots shown in the main text of the manuscript.
	- `run_all_models.sh`:  Bash script that shows how all models can be run.
	- `test_stationarity.py`:  Script to test the assumption of statistical stationary as described in Appendix S3.


