import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import landscape_simulation as land_sim
import glob
import subprocess
import os
import model3_simulation as bl
import importlib
import copy
import yaml
import multiprocessing as mp
import sys
importlib.reload(land_sim)


"""
Script to simulate Model 1, 2 and 3  

To use this script, at the command line type

python3 model0-2_simulation.py model_parameters.yml 

This will run the baseline simulation. See run_all_models.sh.
"""

def run_simulation(ns, init, params, foi_map, t_array, time_length, deltat, external_infection, model_nm, model_summary):
    """
    Multiprocess the running of SEIR simulations
    """

    land_sim.logger.info("Starting simulation {0}".format(ns + 1))

    if model_nm == "model0":

        # Model 1 in the main text
        all_res = land_sim.seir_simulation_discrete_stoch_time_varying_model0(init, params['n'], 
                                          params['sigma'], 
                                          params['gamma'],
                                          params['nu'],
                                          foi_map,
                                          t_array, 
                                          time_length, deltat, external_infection)

    else:
        all_res = land_sim.seir_simulation_discrete_stoch_time_varying(init, params['n'], 
                                          params['sigma'], 
                                          params['gamma'],
                                          params['nu'],
                                          foi_map,
                                          t_array, 
                                          time_length, deltat, external_infection)
    
    seir_traj = all_res.sum(axis=1)
    spatial_stats = land_sim.get_spread_metrics(all_res, model_summary)

    return((seir_traj, spatial_stats))


def build_memory_map(foi):
    """
    Memory map the FOI array
    """

    os.makedirs("tmp/", exist_ok=True)
    foi_map = np.memmap("tmp/foi.dat", dtype="float64", mode="w+", shape=foi.shape)
    foi_map[:, :, :] = foi[:, :, :]
    foi_map.flush()
    foi_map = np.memmap("tmp/foi.dat", dtype="float64", mode="r", shape=foi.shape)
    del foi # Delete the FOI matrix
    return(foi_map)

if __name__ == '__main__':

    ################################
    ##### Set-up the landscape #####
    ################################

    baseline_parameters = "model_parameters.yml"
    parameter_set = sys.argv[1]

    # Load parameters
    with open(baseline_parameters, 'r') as file:
        model_params = yaml.safe_load(file)

    # Load parameters
    with open(parameter_set, 'r') as file:
        update_params = yaml.safe_load(file)

    # Update the baseline parameters
    model_params['parameter_set_name'] = update_params['parameter_set_name']
    for k1 in update_params.keys():
        if type(update_params[k1]) == dict:
            for k2 in update_params[k1].keys():
                model_params[k1][k2] = update_params[k1][k2]

    # Extract the name of the parameter set
    parameter_set_name = model_params['parameter_set_name']

    grid_size = model_params['landscape']['grid_size'] # meters
    buffer = model_params['landscape']['buffer'] # meters
    filenames_for_ud = glob.glob("../results/host_uds/*ud_all_season=Gestation_year*.pkl")
    collar_info_path = "../data/deer_data/collar_data_01252025.csv"

    # Location to store UDs for simulated individuals
    filepath = "./dynamic_sim_uds"

    # Assign sexes to uds
    filenames_for_ud, uds_by_sex = bl.get_uds_by_sex(filenames_for_ud, collar_info_path)

    # Landscape boundary
    bounds = model_params['landscape']['bounds']

    # Specify the group size distribution...this will be dynamic...but start here
    gsize_dist = land_sim.ames_group_size_distributions()
    deer_density = model_params['landscape']['deer_density'] / 1000000 # Scale to meters 
    wrap_landscape = model_params['landscape']['wrap_landscape']
    land = land_sim.Landscape(bounds, grid_size, buffer)

    # Crop polygons and load
    tbounds = land.polygon.bounds
    shp = gpd.read_file("../data/spatial_data/{0}".format(model_params['landscape']['forest_shapefile']))

    # If True, re-build the adjacency matrices
    # Otherwise, load them from disk
    build_adj_matrices = True
    run_epi_simulations = True
    load_bvals = False # Load bvals from disk
    landscape_numbers = np.arange(1, model_params['sim']['num_landscapes'] + 1) 
    seeds = np.array([55, 10, 45, 123, 34, 2, 234, 23, 89, 102, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]) 
    num_years = 3 # Total length of simulation
    year_length = model_params['demography']['year_length']

    # Extract with and between group correlations
    cw = tuple(model_params['demography']['ratio_within_group'])
    cb = tuple(model_params['demography']['ratio_between_group'])

    for landscape_number in landscape_numbers:

        land_sim.logger.info("Beginning Landscape {0}".format(landscape_number))

        if build_adj_matrices:
            ##### Model 1 adjacency matrix: Fixed UDs for the gestation season #####

            # Initialize the landscape
            np.random.seed(seeds[landscape_number - 1]) #np.random.seed(10)
            land.assign_individuals_randomly_to_landscape(deer_density, 
                                                          gsize_dist, 
                                                          number=False,
                                                          polygons=shp.geometry,
                                                          age_structure=np.array(model_params['demography']['age_structure']))

            land_sim.logger.info("Working on model 1...")
            land, adj_matrix1 = bl.season1_landscape(land, filenames_for_ud, uds_by_sex, 
                                                     filepath, cw, cb, wrap_landscape=wrap_landscape)

            model1_positions = land.build_individual_dataframe()

            ##### Model 0 adjacency matrix: Null model, equally connected adjacency matrix #####
            adj_matrix0 = np.repeat(grid_size*grid_size / land.area, len(adj_matrix1.ravel())).reshape(adj_matrix1.shape) # np.ones(adj_matrix1.shape)
            adj_matrix0[np.diag_indices(adj_matrix0.shape[0])] = 0 # Can't infect yourself

            ##### Model 2 adjacency matrix:  #####
            land_sim.logger.info("Working on model 2...")
            model2 = [adj_matrix1]
            model2_positions = [model1_positions]

            # Loop through
            disperse_percents = np.tile([model_params['demography']['percent_male_dispersal_spring'], 
                                           model_params['demography']['percent_male_dispersal_fall']], num_years)
            for i in range(num_years*2):
                land_sim.logger.info("Landscape {0}...".format(i))

                # Reset status of all individuals to not dispersed
                if i % 2 == 0:
                    [ind.set_dispersed(False) for ind in land.individuals]

                land, tadj_matrix = bl.season2_landscape(land, cw, cb,
                                                          habitat_polygons=shp.geometry, 
                                                          percent_disperse=disperse_percents[i], 
                                                          buffer=model_params['demography']['buffer'], 
                                                          group=True, 
                                                          mean_dispersal=model_params['demography']['mean_dispersal'],
                                                          k=model_params['demography']['scale_dispersal'],
                                                          only_males=model_params['demography']['only_males'],
                                                          wrap_landscape=wrap_landscape)

                tpositions = land.build_individual_dataframe()

                model2.append(tadj_matrix)
                model2_positions.append(tpositions)

                # Account for wrapping seasonality around the year
                if i % 2 == 1 and i != (num_years*2 - 1):
                    model2.append(tadj_matrix)
                    model2_positions.append(tpositions)

            ##### Set up the adjacency matrices for the model #####

            duration_of_seasons = model_params['demography']['duration_of_seasons_model2']

            model0 = np.array([adj_matrix0]) # This could probably be simulated really easily with a Gillespie...good to compare
            model0_timing = np.array([year_length*num_years])
            model1 = np.array([adj_matrix1])
            model1_timing = np.array([year_length*num_years])
            model2 = np.array(model2)
            model2_timing = np.tile(duration_of_seasons, num_years)

            model_values = {"model0": [model0, model0_timing, model1_positions],
                            "model1": [model1, model1_timing, model1_positions],
                            "model2": [model2, model2_timing, model2_positions]}

            # Save models
            pd.to_pickle((True, model_values), "/Volumes/MarkWilbersEH/results/matrices_for_simulated_landscape{0}_{1}.pkl".format(landscape_number, parameter_set_name))

        else:

            ps = update_params.get("parameter_set_name_for_landscape", parameter_set_name)
            land, model_values = pd.read_pickle( "/Volumes/MarkWilbersEH/results/matrices_for_simulated_landscape{0}_{1}.pkl".format(landscape_number, ps))

        if run_epi_simulations:

            ### Epi simulations ###
            os.makedirs("../results/stochastic_simulations", exist_ok=True)

            # Set your simulation parameters
            params = {}
            params['sigma'] = model_params['epi']['sigma'] 
            params['gamma'] = model_params['epi']['gamma'] 
            num_sims = model_params['sim']['num_sims']
            external_infection_rate = model_params['epi']['external_infection_rate']
            cores = model_params['sim']['cores'] # Number of multiprocess cores

            time_length = year_length*num_years # days of the simulation

            desired_R0_vals = np.array(model_params['sim']['desired_R0_vals']) # Average number of new infections produced per individual per day
            days_waning_antibody = np.array(model_params['sim']['months_waning_antibody'])*30 # Average number of days (months * 30 days) in recovered class

            # Start loop here for each model
            for model_name in model_values.keys():

                model, model_timing, model_summary = model_values[model_name]
                params['n'] = model.shape[-1]

                # Get your desired beta values given R0 values
                # Note that we are only calculating R0 for the first
                # temporal window so the estimate is more comparable 
                # across models.  Note that R0 > 1 during gestation nearly 
                # gauruntees that R0 > 1 for the season. We have confirmed this
                # by calculating seasonal variants of r.
                if load_bvals:
                    desired_beta_vals = np.empty(len(desired_R0_vals))
                else:

                    print("Computing beta vals from R0 vals...")
                    R0_unnorm, R0, _, _ = land_sim.movement_R0_from_avg_foi(model[0], params['gamma'])
                    desired_beta_vals = desired_R0_vals / R0
                    maxR0 = R0_unnorm.sum(axis=1).argmax()
                    model_params['epi']['starting_index'] = maxR0
                    print("Done")


                # Start loop here for each beta_value corresponding to a desired r
                for j, bval in enumerate(desired_beta_vals):

                    # Build foi matrix and memory map it for more efficient
                    # multiprocessing

                    if load_bvals:
                        R0val = desired_R0_vals[j]
                        _, modp = pd.read_pickle("/Volumes/MarkWilbersEH/results/stochastic_simulations/model_name={0}_R0={1}_days_in_recovery={2}_landscape{3}_{4}.pkl".format(model_name, desired_R0_vals[j], days_waning_antibody[0], landscape_number, parameter_set_name))
                        bval = modp['epi']['beta']
                        maxR0 = modp['epi'].get('starting_index', 20)

                    model_params['epi']['beta'] = bval
                    model_params['epi']['medianR0'] = np.median((R0_unnorm*bval).sum(axis=1))
                    foi_map = build_memory_map(model*bval)

                    # Set the initial conditions
                    init = np.zeros((4, params['n']), dtype=np.int64)
                    init[0, :] = 1
                    init[0, maxR0] = 0
                    init[2, maxR0] = 1

                    deltat = 1.0 # One day time step for the simulation...note that this won't exactly correspond to the beta values because the delta t is different
                    external_infection = np.repeat(external_infection_rate, params['n'])
                    t_array = np.cumsum(model_timing / time_length) # CHECK THIS

                    # Run simulation
                    for d in days_waning_antibody:

                        all_sims = []
                        params['nu'] = 1 / d
                        land_sim.logger.info("Starting model {0} with r = {1} and days waning = {2}".format(model_name, desired_R0_vals[j], d))

                        # Multiprocess simulations
                        pool = mp.Pool(processes=cores)
                        results = [pool.apply_async(run_simulation, args=(i, init, params, foi_map, t_array, 
                                                                         time_length, deltat, external_infection, model_name, model_summary))
                                        for i in range(num_sims)] 
                        all_sims = [p.get() for p in results]
                        pool.close()

                        # Save simulation results for a particular model
                        pd.to_pickle((all_sims, model_params), "/Volumes/MarkWilbersEH/results/stochastic_simulations/model_name={0}_R0={1}_days_in_recovery={2}_landscape{3}_{4}.pkl".format(model_name, desired_R0_vals[j], d, landscape_number, parameter_set_name))


