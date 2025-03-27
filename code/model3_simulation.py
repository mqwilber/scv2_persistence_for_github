import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import landscape_simulation as land_sim
import glob
import subprocess
import os
import importlib
import copy
import scipy.stats as stats
import yaml
import multiprocessing as mp
import sys
importlib.reload(land_sim)

"""
Simulate Model 3 with full seasonality.  The parameters in this script
should match the parameters in model0-2_simulation.py
Model 3 requires a some specific details and is thus implemented in a different
script.
"""

def season1_landscape(land, filenames_for_ud, uds_by_sex, filepath,
                      ratio_within_group, ratio_between_group, 
                      just_social_connections=False,
                      wrap_landscape=False):
    """
    Build the landscape for the gestation time period

    Parameters
    ----------
    land : Landscape object
    filenames_for_ud : string
        Filename where UDs pre-computed and stored
    ud_by_sex : None or array-like
        If array-like, it should specify whether each filename corresponds
        to a male or female
    filepath : string
        Path where UDs will be stored
    ratio_with_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals within a social group
    ratio_between_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals in different social groups
    just_social_connections : bool
        If True, only update social connections. Otherwise, update spatial
        connections
    wrap_landscape : bool
        Wrap the edges of the landscape to approximate a larger landscape.

    Return
    ------
    : tuple
        Landscape object, adjacency matrix

    """

    if not just_social_connections:
        land.assign_individuals_uds_mp(filenames_for_ud, 
                                         ud_storage_path=filepath,
                                         buffer=0, cores=8, 
                                         Z_in_ram=True, randomize=True, 
                                         uds_by_sex=uds_by_sex)
        land.get_adjacency_matrix_split(Z_in_ram=True, wrap_landscape=wrap_landscape,
                                        ratio_within_group=ratio_within_group,
                                        ratio_between_group=ratio_between_group)
    else:

        alive_now = np.array([ind.alive for ind in land.individuals])
        alive_ids = np.where(pd.Series(alive_now))[0]
        land.get_adjacency_matrix_split(Z_in_ram=True,
                                        update=True,    
                                        update_ids=alive_ids, compare_ids=alive_ids,
                                        update_surfaces=['social'],
                                        wrap_landscape=wrap_landscape,
                                        ratio_within_group=ratio_within_group,
                                        ratio_between_group=ratio_between_group)

    adj_matrix1 = np.copy(land.adjacency_matrix)
    return((land, adj_matrix1))


def season2_landscape(land, ratio_within_group, ratio_between_group, 
                      habitat_polygons=None, percent_disperse=1, 
                      buffer=1000, group=True, mean_dispersal=6, k=5,
                      only_males=True, wrap_landscape=False):
    """
    Allow for young buck dispersal in the spring

    Parameters
    ----------
    land : Landscape object
    ratio_with_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals within a social group
    ratio_between_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals in different social groups
    habitat_polygons: GeoDataFrame
        Contains polygons that specify where dispersing males will move to
    percent_disperse : float
        The percent of yearling males that disperse
    buffer : float
        The buffer around dispersal location in which males looks for other
        males to group with
    group : bool
        If True, males will group with other males when they disperse. Otherwise,
        they won't. 
    only_males : bool
        If True, only young males dispersal.  If False, female and males dispers
    wrap_landscape : bool
        Wrap the edges of the landscape to approximate a larger landscape.


    Return
    ------
    : tuple
        Landscape object, adjacency matrix
    """
    
    sexes = np.array([ind.sex for ind in land.individuals])
    ages = np.array([ind.age for ind in land.individuals])
    alive_now = np.array([ind.alive for ind in land.individuals])
    alive_ids = np.where(pd.Series(alive_now))[0]
    not_dispersed = ~np.array([ind.dispersed for ind in land.individuals])

    # Which individuals are dispersing? Yearling males and females
    if only_males:
        male_ids = np.where(pd.Series(sexes == "M") & pd.Series(ages == "Y") & pd.Series(alive_now) & pd.Series(not_dispersed))[0]
    else:
        male_ids = np.where((pd.Series(sexes == "M") | pd.Series(sexes == "F")) & pd.Series(ages == "Y") & pd.Series(alive_now) & pd.Series(not_dispersed))[0]

    # Some percent of young males disperse
    disperse_ids = np.sort(np.random.choice(male_ids, replace=False,
                                    size=np.random.binomial(p=percent_disperse, 
                                                            n=len(male_ids), 
                                                            size=1)[0])) 

    # Let males disperse
    [land.individual_dispersal(ind_id, habitat_polygons=habitat_polygons, 
                                buffer=1000, group=group, 
                                mean_dispersal=mean_dispersal,
                                k=k) for ind_id in disperse_ids]

    # Update adjacency
    land.get_adjacency_matrix_split(Z_in_ram=True, 
                                    update=True, update_ids=disperse_ids,
                                    compare_ids=alive_ids,
                                    wrap_landscape=wrap_landscape,
                                    ratio_within_group=ratio_within_group,
                                    ratio_between_group=ratio_between_group)

    adj_matrix2 = np.copy(land.adjacency_matrix)
    return((land, adj_matrix2))


def season3_landscape(land, repro_percent_adults=0.9, wrap_landscape=False):
    """
    Pregnant females isolate themselves and give birth from June - July.

    Parameters
    ----------
    land : Landscape object
    repro_percent_adults : float
        Probability that an adult or yearling is pregnant
    wrap_landscape : bool
        Wrap the edges of the landscape to approximate a larger landscape.

    Return
    ------
    : tuple
        Landscape object, adjacency matrix
    """

    sexes = np.array([ind.sex for ind in land.individuals])
    ages = np.array([ind.age for ind in land.individuals])
    alive_now = np.array([ind.alive for ind in land.individuals])
    alive_ids = np.where(pd.Series(alive_now))[0]

    # Add an age criteria here...assume fawns don't breed...but it looks like they do
    female_ids = np.where(pd.Series(sexes == "F") & pd.Series(ages != "F") & pd.Series(alive_now))[0]

    reproduce_ids = np.sort(np.random.choice(female_ids, replace=False,
                                    size=np.random.binomial(p=repro_percent_adults, n=len(female_ids), size=1)[0])) 

    # Update adjacency matrix to remove interactions for fawning females as
    # they isolate themselves. They can still have spatial interactions.
    # Just no social interactions
    land.get_adjacency_matrix_split(Z_in_ram=True,
                              update=True, update_ids=reproduce_ids,
                              compare_ids=alive_ids,
                              update_surfaces=['social'], 
                              fixed_correlation=True, ratio_val=-2,
                              wrap_landscape=wrap_landscape)

    adj_matrix3 = np.copy(land.adjacency_matrix)

    return((land, adj_matrix3, reproduce_ids))


def season4_landscape(land, reproduce_ids, 
                      filenames_for_ud,
                      uds_by_sex, filepath, birth_time,
                      ratio_within_group, ratio_between_group,
                      neonate_survival=0.2, expected_neonates=2,
                      wrap_landscape=False):
    """
    Fawns recruit into the population and moms re-establish social connections
    with the group

    Parameters
    ----------
    reproduce_ids : array-like
        The ids of females that were pregnant
    filenames_for_ud : string
        Path to UDs to assign to fawns
    uds_by_sex : array-like
        Specifies which uds correspond to which sex
    filepath : string
        Path to store UDs
    birth_time : float
        Specifies when the individuals were born
    ratio_with_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals within a social group
    ratio_between_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals in different social groups
    neonate_survival : float
        A probability that neonates survive birth and recruit
    expected_neonates : float
        The expected number of neonates per female
    wrap_landscape : bool
        Wrap the edges of the landscape to approximate a larger landscape.
    """

    # Update the ages of individuals
    for ind in land.individuals:

        if ind.age == 'F':
            # Fawns to yearling 
            ind.age = 'Y'
        elif ind.age == 'Y':
            # Yearlings to adults
            ind.age = 'A'
        else:
            # Adults stay as adults
            pass

    alive_now = np.array([ind.alive for ind in land.individuals])
    alive_ids = np.where(pd.Series(alive_now))[0]

    # How many babies are moms having that mature to fawns?
    alive_reproduce = np.array([land.individuals[ri].alive for ri in reproduce_ids])

    reproduce_ids_alive = reproduce_ids[alive_reproduce]
    fawns = np.random.poisson(neonate_survival*expected_neonates, size=len(reproduce_ids_alive))

    mom_ids = [land.individuals[i] for i in reproduce_ids_alive]

    # Assign the fawns to the landscape
    [[land.add_individual_to_landscape(m_id, birth_time) for j in range(babies)] 
                                        for babies, m_id in zip(fawns, mom_ids)]


    # Assign the new fawns UDs
    add_fawns = land.individuals[-np.sum(fawns):]
    land.assign_individuals_uds_mp(filenames_for_ud, 
                                     ud_storage_path=filepath,
                                     buffer=0, cores=8,
                                     Z_in_ram=True, randomize=True, 
                                     uds_by_sex=uds_by_sex,
                                     specific_individuals=add_fawns)

    # Update adjaceny matrix to include fawn interactions
    fawn_ids = [ind.individual_id for ind in add_fawns]

    # First append all the fawns to the adjacency matrix
    land.get_adjacency_matrix_split(Z_in_ram=True, 
                                    update=True, 
                                    update_ids=fawn_ids,
                                    compare_ids=alive_ids,
                                    append_adj=True,
                                    wrap_landscape=wrap_landscape,
                                    ratio_within_group=ratio_within_group,
                                    ratio_between_group=ratio_between_group)

    # Restablish the female social connections
    land.get_adjacency_matrix_split(Z_in_ram=True, 
                              update=True, 
                              update_ids=reproduce_ids_alive,
                              compare_ids=alive_ids,
                              update_surfaces=['social'],
                              wrap_landscape=wrap_landscape,
                              ratio_within_group=ratio_within_group,
                              ratio_between_group=ratio_between_group)

    adj_matrix4 = np.copy(land.adjacency_matrix)
    return((land, adj_matrix4, fawn_ids))


def season5_landscape(land, ratio_within_group,
                            ratio_between_group,
                            strength_of_mating_interaction=2,
                            habitat_polygons=None,
                            percent_disperse=1,
                            mean_dispersal=6, k=5,
                            only_males=True,
                            wrap_landscape=False):
    """
    During the rut, young males disperse, males break social connections and 
    males and females build social connections.

    Parameters
    ----------
    ratio_with_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals within a social group
    ratio_between_group : tuple
        Tuple, on the log10 scale, giving the ratio of social to spatial
        contributions to FOI for individuals in different social groups
    strength_of_mating_interaction : float
        The log10 ratio of correlation to spatial overlap for male and females during rut
    habitat_polygons : GeoDataFrame
        Specifies the polygons where young males can disperse
    percent_disperse : float
        The percent of young males that disperse
    only_males : bool
        If True, only yearling males disperse.  If False, yearling males and females disperse
    wrap_landscape : bool
        Wrap the edges of the landscape to approximate a larger landscape.
    """

    sexes = np.array([ind.sex for ind in land.individuals])
    ages = np.array([ind.age for ind in land.individuals])
    alive_now = np.array([ind.alive for ind in land.individuals])
    alive_ids = np.where(pd.Series(alive_now))[0]
    not_dispersed = ~np.array([ind.dispersed for ind in land.individuals])

    # Which individuals are dispersing? Yearling males and females
    if only_males:
        male_ids = np.where(pd.Series(sexes == "M") & pd.Series(ages == "Y") & pd.Series(alive_now) & pd.Series(not_dispersed))[0]
    else:
        male_ids = np.where((pd.Series(sexes == "M") | pd.Series(sexes == "F")) & pd.Series(ages == "Y") & pd.Series(alive_now) & pd.Series(not_dispersed))[0]


    # Some percent of young males disperse again in the fall
    disperse_ids = np.sort(np.random.choice(male_ids, replace=False,
                                    size=np.random.binomial(p=percent_disperse, 
                                                            n=len(male_ids), 
                                                            size=1)[0])) 

    # Let males disperse and find a new group (they will lose this correlation in the next step)
    [land.individual_dispersal(ind_id, habitat_polygons=habitat_polygons, 
                                buffer=1000, group=True,
                                mean_dispersal=mean_dispersal, k=k) for ind_id in disperse_ids]
    land.get_adjacency_matrix_split(Z_in_ram=True, 
                                    update=True, 
                                    update_ids=disperse_ids,
                                    compare_ids=alive_ids,
                                    wrap_landscape=wrap_landscape,
                                    ratio_within_group=ratio_within_group,
                                    ratio_between_group=ratio_between_group)


    # Males drop social connections with other males to look for females
    sexes = np.array([ind.sex for ind in land.individuals])
    male_ids = np.where(pd.Series(sexes == "M") & pd.Series(alive_now))[0] # Add age-specific criteria here

    land.get_adjacency_matrix_split(Z_in_ram=True, 
                                    update=True, update_ids=male_ids, 
                                    compare_ids=male_ids,
                                    fixed_correlation=True, ratio_val=-2,
                                    update_surfaces=['social'])

    # Males build social connections with females for mating
    female_ids = np.where(pd.Series(sexes == "F") & pd.Series(alive_now))[0]
    land.get_adjacency_matrix_split(Z_in_ram=True,
                              update=True, update_ids=male_ids, 
                              compare_ids=female_ids,
                              fixed_correlation=True,
                              ratio_val=strength_of_mating_interaction,
                              update_surfaces=['social'])

    adj_matrix5 = np.copy(land.adjacency_matrix)
    return((land, adj_matrix5))


def augment_matrix(mat, desired_shape):
    """
    Buffer a matrix with zeros
    """
    old_shape = mat.shape

    new_mat = np.zeros(desired_shape)
    new_mat[:old_shape[0], :old_shape[1]] = mat
    return(new_mat)


def get_uds_by_sex(filenames_for_ud, collar_info_path):
    """
    For A list of filenames, get the sex of the animal associated with the 
    filename

    Parameters
    ----------
    filenames_for_ud : list
        List of filenames that contain UDs
    collar_info_path : string
        A path to the file that links ID and sex for the deer. It should
        have columns, 'Collar_ID_Iridium' and 'Sex'.

    """

    deer_ids = np.array([os.path.basename(fnm).split("_")[0] for fnm in filenames_for_ud]).astype(np.int64)
    all_ids = pd.DataFrame({'deer_id': deer_ids, 'fname': filenames_for_ud})

    # Get ID by sex information
    dat = pd.read_csv(collar_info_path)[['Collar_ID_Iridium', 'Sex']].dropna()
    dat = dat.rename(columns={'Collar_ID_Iridium': "deer_id", "Sex": "sex"})
    temp = all_ids.set_index('deer_id').join(dat.set_index("deer_id"))
    filenames_for_ud = temp.fname.values
    uds_by_sex = temp.sex.values

    return((filenames_for_ud, uds_by_sex))


def run_simulation(ns, init, params, foi, t_array, time_length, deltat, 
                   external_infection, death_times, birth_times, model_summary):
    """
    Multiprocess the running of SEIR simulations.  See land_sim.seir_simulation_discrete_stoch_time_varying_birth_death
    for arguments.
    """

    land_sim.logger.info("Starting simulation {0}".format(ns + 1))
    all_res = land_sim.seir_simulation_discrete_stoch_time_varying_birth_death(init, params['n'], 
                                      params['sigma'], 
                                      params['gamma'],
                                      params['nu'],
                                      foi,
                                      t_array, 
                                      time_length, deltat, external_infection,
                                      death_times, birth_times)
    seir_traj = all_res.sum(axis=1)
    spatial_stats = land_sim.get_spread_metrics(all_res, model_summary)

    return((seir_traj, spatial_stats))


def run_simulation_less_memory(ns, init, params, foi, t_array, time_length, deltat, 
                   external_infection, death_times, birth_times, model_summary):
    """
    Multiprocess the running of SEIR simulations.  See land_sim.seir_simulation_discrete_stoch_time_varying_birth_death
    for arguments.

    This version of the simulation uses less memory, but does not compute the spatial
    statistics
    """

    land_sim.logger.info("Starting simulation {0}".format(ns + 1))
    all_res = land_sim.seir_simulation_discrete_stoch_time_varying_birth_death_less_memory(init, params['n'], 
                                      params['sigma'], 
                                      params['gamma'],
                                      params['nu'],
                                      foi,
                                      t_array, 
                                      time_length, deltat, external_infection,
                                      death_times, birth_times)
    seir_traj = np.empty((6, all_res.shape[1]))
    for i in range(seir_traj.shape[1]):
        seir_traj[:, i] = np.bincount(all_res[:, i], minlength=6)

    # seir_traj = all_res.sum(axis=1)
    # spatial_stats = land_sim.get_spread_metrics(all_res, model_summary)
    return((seir_traj, None))

    # return((seir_traj, spatial_stats))



def update_dead_alive_for_model(model, model_positions):
    """
    For calculating r or R0, zero out foi of dead/unborn individuals for each matrix
    """

    new_model = []

    for i in range(len(model)):
        tm = model[i]
        alive = model_positions[i].alive
        alive_full = np.hstack([alive, np.repeat(0, tm.shape[0] - len(alive))])
        tred = ((tm * alive_full).T * alive_full).T
        new_model.append(tred)

    new_model = np.array(new_model)
    return(new_model)


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

    baseline_parameters = "model_parameters.yml"
    parameter_set = sys.argv[1] #"model_parameters_all_yearlings_disperse.yml"

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

    ### Set-up the landscape ###

    grid_size = model_params['landscape']['grid_size'] # meters
    buffer = model_params['landscape']['buffer'] # meters
    wrap_landscape = model_params['landscape']['wrap_landscape'] # Is the landscape wrapped?
    filenames_for_ud = glob.glob("../results/host_uds/*ud_all_season=Gestation_year*.pkl")
    collar_info_path = "../data/deer_data/collar_data_01252025.csv"

    # Location to store UDs for simulated individuals
    filepath = "./dynamic_sim_uds"

    # Assign sexes to uds
    filenames_for_ud, uds_by_sex = get_uds_by_sex(filenames_for_ud, collar_info_path)

    # Ames bounds
    bounds = model_params['landscape']['bounds']

    ames = gpd.read_file("../data/spatial_data/ames boundary.shp")

    # Specify the group size distribution...this will be dynamic...but start here
    gsize_dist = land_sim.ames_group_size_distributions()
    deer_density = model_params['landscape']['deer_density'] / 1000000 # Convert to meters

    land = land_sim.Landscape(bounds, grid_size, buffer)

    # Crop polygons and load
    tbounds = land.polygon.bounds
    shp = gpd.read_file("../data/spatial_data/{0}".format(model_params['landscape']['forest_shapefile']))

    # Initialize the landscape
    build_adj_matrices = True
    run_epi_simulations = True
    less_memory = False # Set to true if you need to run a simulation with less memory. Doesn't calculate spatial results.
    landscape_numbers = np.arange(1, model_params['sim']['num_landscapes'] + 1)
    seeds = np.array([55, 10, 45, 123, 34, 2, 234, 23, 89, 102, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    duration_of_seasons = model_params['demography']['duration_of_seasons_model3']
    year_length = model_params['demography']['year_length']
    total_year = np.sum(duration_of_seasons)
    num_years = 3

    # Death rate parameters for age class by sex
    death_rates = {'F, M': -1*np.log(model_params['demography']['fawn_male_survival']) / 365,
                   'F, F': -1*np.log(model_params['demography']['fawn_female_survival']) / 365,
                   'Y, M': -1*np.log(model_params['demography']['yearling_male_survival']) / 365,
                   'Y, F': -1*np.log(model_params['demography']['yearling_female_survival']) / 365,
                   'A, M': -1*np.log(model_params['demography']['adult_male_survival']) / 365, 
                   'A, F': -1*np.log(model_params['demography']['adult_female_survival']) / 365}

    # Extract with and between group correlations
    cw = tuple(model_params['demography']['ratio_within_group'])
    cb = tuple(model_params['demography']['ratio_between_group'])

    for landscape_number in landscape_numbers:

        land_sim.logger.info("Beginning Landscape {0}".format(landscape_number))

        if build_adj_matrices:

            np.random.seed(seeds[landscape_number - 1])
            land.assign_individuals_randomly_to_landscape(deer_density, 
                                                          gsize_dist, 
                                                          number=False,
                                                          polygons=shp.geometry,
                                                          age_structure=np.array(model_params['demography']['age_structure']))
            # All individuals were born before 0, so set to 0
            [ind.set_birth_time(0) for ind in land.individuals]

            season1_mats = []
            season2_mats = []
            season3_mats = []
            season4_mats = []
            season5_mats = []
            model3_positions = []

            for t in range(num_years):

                ### Season 1: February, Gestation 
                if t == 0:
                    # Build the full landscape
                    land_sim.logger.info("Working on season 1...")
                    land, adj_matrix1 = season1_landscape(land, filenames_for_ud, uds_by_sex, 
                                                          filepath, cw, cb, 
                                                          just_social_connections=False,
                                                          wrap_landscape=wrap_landscape)
                else:
                    land_sim.logger.info("Working on season 1...")
                    land, adj_matrix1 = season1_landscape(land, filenames_for_ud, uds_by_sex, 
                                                          filepath, cw, cb,
                                                          just_social_connections=True,
                                                          wrap_landscape=wrap_landscape)


                season1_mats.append(adj_matrix1)
                model3_positions.append(land.build_individual_dataframe())

                land.kill_individuals(duration_of_seasons[0], 
                                      total_year*t + np.sum(duration_of_seasons[:0]),
                                      death_rates)

                ### Season 2: Spring, male dispersal
                land_sim.logger.info("Working on season 2...")
                land, adj_matrix2 = season2_landscape(land, cw, cb, 
                                                        habitat_polygons=shp.geometry, 
                                                        percent_disperse=model_params['demography']['percent_male_dispersal_spring'], 
                                                        buffer=1000, 
                                                        group=True, 
                                                        mean_dispersal=model_params['demography']['mean_dispersal'],
                                                        k=model_params['demography']['scale_dispersal'],
                                                        only_males=model_params['demography']['only_males'],
                                                        wrap_landscape=wrap_landscape)
                season2_mats.append(adj_matrix2)
                model3_positions.append(land.build_individual_dataframe())

                land.kill_individuals(duration_of_seasons[1], 
                                      total_year*t + np.sum(duration_of_seasons[:1]),
                                      death_rates)

                ### Season 3: Female fawning and isolation
                land_sim.logger.info("Working on season 3...")
                land, adj_matrix3, reproduce_ids = season3_landscape(land, 
                                                                     repro_percent_adults=model_params['demography']['repro_percent_adults_yearlings'],
                                                                     wrap_landscape=wrap_landscape)
                season3_mats.append(adj_matrix3)
                model3_positions.append(land.build_individual_dataframe())

                land.kill_individuals(duration_of_seasons[2], 
                                      total_year*t + np.sum(duration_of_seasons[:2]),
                                      death_rates)

                ### Season 4: Neonates recruit and females rebuild social interactions
                land_sim.logger.info("Working on season 4...")
                birth_time = np.sum(duration_of_seasons[:3])
                land, adj_matrix4, fawn_ids = season4_landscape(land, reproduce_ids, 
                                                                filenames_for_ud,
                                                                uds_by_sex, 
                                                                filepath,
                                                                birth_time + total_year*t,
                                                                cw, cb,
                                                                neonate_survival=model_params['demography']['neonate_survival'], 
                                                                expected_neonates=model_params['demography']['expected_neonates'],
                                                                wrap_landscape=wrap_landscape)
                season4_mats.append(adj_matrix4)
                model3_positions.append(land.build_individual_dataframe())


                land.kill_individuals(duration_of_seasons[3], 
                                      total_year*t + np.sum(duration_of_seasons[:3]),
                                      death_rates)


                #### Season 5: Rut, males disperse, dissassociate and find females
                land_sim.logger.info("Working on season 5...")
                land, adj_matrix5 = season5_landscape(land, cw, cb,
                                                      strength_of_mating_interaction=model_params['demography']['strength_of_mating_interaction'],
                                                      habitat_polygons=shp.geometry,
                                                      percent_disperse=model_params['demography']['percent_male_dispersal_fall'],
                                                      mean_dispersal=model_params['demography']['mean_dispersal'],
                                                      k=model_params['demography']['scale_dispersal'],
                                                      only_males=model_params['demography']['only_males'],
                                                      wrap_landscape=wrap_landscape)

                season5_mats.append(adj_matrix5)
                model3_positions.append(land.build_individual_dataframe())

                land.kill_individuals(duration_of_seasons[4], 
                                      total_year*t + np.sum(duration_of_seasons[:4]),
                                      death_rates)

                # Append one final data frame with dead individuals
                if t == np.max(range(num_years)):
                    model3_positions.append(land.build_individual_dataframe())

            # Augment all of the matrices to be the same shape
            desired_shape = season5_mats[-1].shape
            season1_mats = [augment_matrix(mat, desired_shape) for mat in season1_mats]
            season2_mats = [augment_matrix(mat, desired_shape) for mat in season2_mats]
            season3_mats = [augment_matrix(mat, desired_shape) for mat in season3_mats]
            season4_mats = [augment_matrix(mat, desired_shape) for mat in season4_mats]
            season5_mats = [augment_matrix(mat, desired_shape) for mat in season5_mats]

            # Set-up the array of matrices
            model3 = np.array([season1_mats[0], season2_mats[0], season3_mats[0], season4_mats[0], season5_mats[0],
                               season1_mats[1], season2_mats[1], season3_mats[1], season4_mats[1], season5_mats[1],
                               season1_mats[2], season2_mats[2], season3_mats[2], season4_mats[2], season5_mats[2]])

            model3_timing = np.tile(duration_of_seasons, 3)

            model_values = {'model3': [model3, model3_timing, model3_positions]}

            # Save the epidemiological landscape
            # True is just a place holder in case you want to save the landscape which is very large
            pd.to_pickle((True, model_values), "/Volumes/MarkWilbersEH/results/matrices_for_simulated_landscape{0}_model3_{1}.pkl".format(landscape_number, parameter_set_name))


        else:

            ps = update_params.get("parameter_set_name_for_landscape", parameter_set_name)
            _, model_values = pd.read_pickle("/Volumes/MarkWilbersEH/results/matrices_for_simulated_landscape{0}_model3_{1}.pkl".format(landscape_number, ps))


        if run_epi_simulations:

            ### Epi simulations ###
            os.makedirs("../results/stochastic_simulations", exist_ok=True)

            model3_positions = model_values['model3'][2]

            # Extract death times and birth times
            death_times = model3_positions[-1].death_time.values 
            birth_times = model3_positions[-1].birth_time.values
            alive_now = model3_positions[-1].alive.values 

            # Set your simulation parameters
            params = {}
            params['sigma'] = model_params['epi']['sigma']
            params['gamma'] = model_params['epi']['gamma'] 
            num_sims = model_params['sim']['num_sims']
            external_infection_rate = model_params['epi']['external_infection_rate']
            cores = 1 #model_params['sim']['cores'] # Number of multiprocess cores

            time_length = year_length*num_years # days of the simulation

            desired_R0_vals = np.array(model_params['sim']['desired_R0_vals']) # Average number of new infections produced per individual per day
            days_waning_antibody = np.array(model_params['sim']['months_waning_antibody'])*30 # Average number of days (months * 30 days) in recovered class

            # Start loop here for each model
            for model_name in model_values.keys():

                model, model_timing, model_summary = model_values[model_name]
                params['n'] = model.shape[-1]

                #  Individuals are born and die. I need
                # to zero out unborn and dead individuals.  I will just do it
                # at each season to keep it simpler.
                new_model = update_dead_alive_for_model(model, model_summary)

                # Get your desired beta values
                print("Computing beta vals from R0 vals...")
                R0_unnorm, R0, _, _ = land_sim.movement_R0_from_avg_foi(new_model[0], params['gamma'])
                desired_beta_vals = desired_R0_vals / R0 # Using some nice properties of R0 to compute this.
                maxR0 = R0_unnorm.sum(axis=1).argmax()
                model_params['epi']['starting_index'] = maxR0
                print("Done")

                # Clean up before the loops
                model_map = build_memory_map(model)
                del model
                del new_model

                # Start loop here for each beta_value corresponding to a desired r
                for j, bval in enumerate(desired_beta_vals):

                    # Build foi matrix.  Yes, this should still be model and not new_model
                    # because the the simulation will account for death appropriately
                    model_params['epi']['beta'] = bval
                    model_params['epi']['medianR0'] = np.median((R0_unnorm*bval).sum(axis=1))
                    foi_map = model_map * bval

                    # Set the initial conditions
                    unborn = birth_times > 0
                    init = np.zeros((6, params['n']), dtype=np.int64)
                    init[0, :] = 1
                    # Single infected individual...MAKE SURE INDIVIDUAL IS BORN! 
                    init[0, maxR0] = 0
                    init[2, maxR0] = 1
                    # Set unborn
                    init[0, unborn] = 0
                    init[4, unborn] = 1

                    deltat = 1.0 
                    external_infection = np.repeat(external_infection_rate, params['n'])
                    t_array = np.cumsum(model_timing / time_length)

                    # Run simulation
                    for d in days_waning_antibody:

                        all_sims = []
                        params['nu'] = 1 / d
                        land_sim.logger.info("Starting model {0} with R0 = {1} and days waning = {2}".format(model_name, desired_R0_vals[j], d))

                        if not less_memory:

                            # Multiprocess simulations
                            pool = mp.Pool(processes=cores)
                            results = [pool.apply_async(run_simulation, args=(i, init, params, foi_map, t_array, 
                                                                             time_length, deltat, external_infection,
                                                                             death_times, birth_times, model_summary))
                                            for i in range(num_sims)] 

                            all_sims = [p.get() for p in results]
                            pool.close()

                        else:

                            all_sims = [run_simulation_less_memory(i, init, params, foi_map, t_array, 
                                                                             time_length, deltat, external_infection,
                                                                             death_times, birth_times, model_summary) for i in range(num_sims)]

                        # Save simulation results for a particular model
                        pd.to_pickle((all_sims, model_params), "/Volumes/MarkWilbersEH/results/stochastic_simulations/model_name={0}_R0={1}_days_in_recovery={2}_landscape{3}_{4}.pkl".format(model_name, desired_R0_vals[j], d, landscape_number, parameter_set_name))







