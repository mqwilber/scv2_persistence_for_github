import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import importlib
import pmovestir as pmstir
import scipy.stats as stats
import itertools
import multiprocessing as mp
import logging
import landscape_simulation as land_sim

logging.basicConfig(filename='process_ud.log', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Initializing logger")


"""
For all deer in the dataset, estimate and save the UDs
"""


def load_and_clean_data(filename, time_step_in_minutes=30):
	"""
	Load and clean deer data, dropping the first 15 days and last 2 days
	"""

	dat = pd.read_csv(filename)
	dat = dat.query("fast_step_ == False and fast_roundtrip_ == False")

	# Trim the movement 
	host_ids = dat.animal_id.unique()

	sorted_hosts = []
	for hid in host_ids:
	    
	    # Remove approximately the first 15 days and the last 2 days
	    tdat = (dat.query("animal_id == {0}".format(hid))
	                    .sort_values(by="t_").iloc[48*15:-48*2, :]
	                    .assign(datetime = lambda x: pd.to_datetime(x.t_))
	                    .assign(unix_time = lambda x: x.datetime.astype(np.int64) / (60 * 10**9)) # Convert nanoseconds to mintues
	                    .rename(columns={'x_': 'UTMx', 'y_': 'UTMy', 
	                                     'animal_id': "individual_ID"}))
	    
	    sorted_hosts.append(tdat)

	# Removing the first 15 days removes 10 animals
	trun_dat = pd.concat(sorted_hosts)

	# Interpolate the trajectories
	deer_dat = trun_dat[['individual_ID', 'datetime', 'unix_time', 'UTMx', 'UTMy']]
	trajs = pmstir.interpolate_all_trajectories(time_step_in_minutes, deer_dat)

	### Preparing data for UD estimation ### 

	# Map the seasons to relevant deer seasons
	seasons_map = {1: "Rut", 2: "Gestation", 3: "Gestation", 4: "Gestation", 5: "Gestation",
	               6: "Fawning", 7: "Fawning", 8: "Lactation",
	               9: "Lactation", 10: "Lactation", 11: "Rut", 12: "Rut"}

	
	deer_trajs = (pd.concat(trajs)
	                .reset_index()
	                .drop(columns="level_1")
	                .rename(columns={"level_0": "individual_ID"})
	                .assign(datetime=lambda x: pd.to_datetime(x.time*60, unit="s"))
	                .assign(month=lambda x: x.datetime.dt.month)
	                .assign(season=lambda x: x.month.map(seasons_map),
	                		year=lambda x: x.datetime.dt.year))

	# Get a season-year columns accounting fo the wrapping of the Rut
	tyears = deer_trajs.year.values
	tyears[deer_trajs.month.values == 1] = tyears[deer_trajs.month.values == 1] - 1 # e.g., Rut in January 2025 is part of 2024 rut
	deer_trajs = deer_trajs.assign(year = tyears)

	return(deer_trajs)



def process_ud_mp(j, host_season_year, deer_trajs, thin, step, logger, total):
	"""
	Calculate The UD for different hosts in different seasons in different years
	"""

	logger.info("Working on {0} of {1}".format(j + 1, total))

	host, season, year = host_season_year.split("_")

	host_locs = deer_trajs.query("individual_ID == {0} and season == '{1}' and year == {2}".format(host, season, np.int64(year)))[["x", "y"]].values.T
	bounds = (np.min(host_locs[0, :]), np.max(host_locs[0, :]), 
			  np.min(host_locs[1, :]), np.max(host_locs[1, :]))

	# Calculate the individual UDs
	host_ud = pmstir.calculate_ud_on_grid(host_locs[:, ::thin], bounds, step)
	pd.to_pickle(host_ud, "../results/host_uds/{0}_ud_all_season={1}_year={2}.pkl".format(host, season, year))

	return((j, "Completed"))


if __name__ == '__main__':
	

	### Load data and data cleaning ###
	filename = "../data/deer_data/cleaned_movement_data_01252025.csv"
	deer_trajs = load_and_clean_data(filename)

	# Specify the hosts to include
	seasons = deer_trajs.season.unique()

	# Identify host, season, year combinations to calculate UDs 
	keep_hosts = (deer_trajs.groupby(["individual_ID", "season", "year"])
			   					   .agg({'x' : len, 'time': lambda x: max(x) - min(x)})
			   					   .reset_index()
			   					   .query("x >= 340"))

	# Days
	stats.scoreatpercentile(keep_hosts.time.values / (60 * 24), (0, 2.5, 25, 50, 75, 97.5, 100))

	host_season_year_ids = (keep_hosts.individual_ID.astype(np.str_) + "_" + 
							keep_hosts.season + "_" + 
							keep_hosts.year.astype(np.str_)).values

	step = 10 # 10 meter grid cells
	thin = 10 # Steps to thin the trajectories (thin every 5 hours to remove autocorrelation)

	# Multiprocess the calculation of the UDs
	pool = mp.Pool(processes=8)
	results = [pool.apply_async(process_ud_mp, 
					args=(j, host, deer_trajs, thin, step, logger, len(host_season_year_ids)))
					for j, host in enumerate(host_season_year_ids)]
	results = [p.get() for p in results]
	pool.close()

	# temp = pd.read_pickle("/Users/mqwilber/Repos/universal_movestir/results/host_uds/151597_ud_all_season=Fawning_year=2023.pkl")
	logger.info("Completed process")




