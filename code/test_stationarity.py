import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.spatial import ConvexHull
import scipy.stats as stats
import os
import pmovestir as pmstir
import generate_UDs as gu
import importlib
from numba import njit


"""
Script to confirm that using a stationary utilization distribution is sufficient
for predicting pairwise infection risk over a 24 hour period.  This script
generates the plots in Appendix S3.
"""

@njit
def get_grid_from_position(xy, xlower, xupper, ylower, yupper):
    """
    Given an x and y position, return the grid indices
    """
    x, y = xy
    xloc = np.where(np.bitwise_and((x > xlower), (x < xupper)))[0][0]
    yloc = np.where(np.bitwise_and((y > ylower), (y < yupper)))[0][0]
    position = np.array([xloc, yloc])
    return(position)


def set_up_movement_model(drift, sigma, mu_center, rho, deltat):
    """
    Set-up the multivariate normal movement model

    Parameters
    ----------
    drift : float
        The drift parameter
    sigma : float
        The variability
    mu_center : array-like
        The home range centers of each pair
    rho : float
        The correlation between the pairs
    deltat : float
        The time step

    Returns
    -------
    : A multivariate-normal distribution

    """

    B = np.diag([drift, drift, drift, drift])
    
    # Diffusion coefficients...m2 per hour, diffusion near the home range
    D_diag = np.diag(np.repeat(sigma, 4))
    
    cormat = np.array([[1, 0, rho, 0],
                       [0, 1, 0, rho],
                       [rho, 0, 1, 0],
                       [0, rho, 0, 1]])
    
    D = D_diag @ cormat @ D_diag # Units of m2 per hour
    Dm = D 
    cov_mat = Dm - expm(-1*B*deltat) @ Dm @ expm(-1*B.T*deltat)
    
    # Stationary distribution
    mvn_stationary = multivariate_normal(mean=mu_center.ravel(), cov=Dm)
    return((mvn_stationary, cov_mat, B))


@njit
def run_simulation(steps, initial_locations, xlower, xupper, ylower, yupper, gamma, 
                   deltat, beta, chol, expmB):

    # Holds host states for two hosts
    host_states = np.empty((2, steps + 1))

    # Host locations. (host 1 x, host 1 y, host 2 x, host 2 y)
    host_locations = np.empty((4, steps + 1))
    # hosts_together = np.empty(steps)

    # Set initial states. Host 1 is always infected
    host_states[0, 0] = 1 # Start infected
    host_states[1, 0] = 0 # Not infected

        # Set initial host locations 
    host_locations[:, 0] = initial_locations

    # Run simulation
    for i in range(steps):

        # Get current host locations
        host1_loc = host_locations[:2, i]
        host2_loc = host_locations[2:, i]

        # Convert to grid to figure out if they are in the same transmission cell
        host1_grid = get_grid_from_position(host1_loc, xlower, xupper, ylower, yupper)
        host2_grid = get_grid_from_position(host2_loc, xlower, xupper, ylower, yupper)

        # hosts_together[i] = np.all(host1_grid == host2_grid)

        # Host 0 can lose infection with some probability each time step
        if host_states[0, i] == 1:
            host_states[0, i + 1] = np.random.binomial(n=1, p=np.exp(-gamma*deltat)) # Don't lose infection
        else:
            host_states[0, i + 1] = 0

        # Epi update
        if np.all(host1_grid == host2_grid):
            
            # They are in the same cell...does infection happen?
            if host_states[1, i] == 1:
                host_states[1, i + 1] = 1 # Stay infected
            else:
                # Gain infection with some probability
                host_states[1, i + 1] =  np.random.binomial(n=1, p=1 - np.exp(-host_states[0, i]*beta*deltat))

        else:
            # Not in the same cell
            
            host_states[1, i + 1] = host_states[1, i]

        # Position update
        xnow = host_locations[:, i][:, np.newaxis]
        # mu = mu_center + expm(-1*B*deltat)@(xnow - mu_center)
        mu = mu_center + expmB @ (xnow - mu_center) # This works because we have a diagnonal matrix

        # host_locations[:, i + 1] = np.random.multivariate_normal(mu.ravel(), cov_mat, size=1).ravel()
        # Use the cholesky decompisition
        host_locations[:, i + 1] = mu.ravel() + chol @ np.random.normal(loc=0, scale=1, size=4)

    return((host_states, host_locations))



if __name__ == '__main__':


    ## Set-up time steps
    deltat = 1 # 1 minute time steps 
    hours = 24 # Length of the simulation in time
    steps = np.int64(hours * (1 / deltat)) # Convert to hours

    ## Set-up movement model
    step_size_vals = np.array([0.001, 0.01, 0.1, 1, 10])
    sigma = 100 # Specifies the area of the home range
    mu_center = np.array([0, 0, 0, 0])[:, np.newaxis]
    rho = 0.8

    ## Set-up epi parameters
    grid = 10 # Grid size in meters determining the area of transmission
    beta = 10 # 0.1 #10 / (grid * grid) # per hour, within area transmission rate
    gamma = 0 #1 / (6 * 24) # per hour, recovery rate 7 days

    ## Set-up landscape
    minx = -550
    maxx = 550
    miny = -550
    maxy = 550
    
    xvals = np.arange(minx, maxx + grid, step=grid)
    yvals = np.arange(minx, maxx + grid, step=grid)
    
    # Get bounds
    xlower = xvals - grid*0.5
    xupper = xvals + grid*0.5
    ylower = yvals - grid*0.5
    yupper = yvals + grid*0.5

    # Number of simulations
    sims = 20000

    # Arrays to hold results
    all_step_sizes = np.empty(len(step_size_vals))
    all_inf_risk = np.empty(len(step_size_vals))
    stationary_inf_risk = np.empty(len(step_size_vals))

    # Loop through drift values
    for d in range(len(step_size_vals)):

        print("Working on d value {0}".format(d))

        ### Set-up OU process and stationary distribution to draw from
        drift = step_size_vals[d]
        print(drift)
        mvn_stationary, cov_mat, B = set_up_movement_model(drift, sigma, mu_center, rho, deltat)
        chol = np.linalg.cholesky(cov_mat)
        expmB = expm(-1*B*deltat)

        # Run internal simulation
        host2_end_state = np.empty(sims)
        mean_step_size = np.empty(sims)
        # hosts_together_all_sims = np.empty((sims, steps))
        
        for s in range(sims):

            initial_locations = mvn_stationary.rvs()
            host_states, host_locations = run_simulation(steps, initial_locations, xlower, xupper, ylower, yupper, gamma, deltat, beta, chol, expmB)
        
        
            # Get mean step size
            total_distance1 = np.mean([pmstir.distance(host_locations[:2, i - 1], host_locations[:2, i]) for i in range(1, host_locations.shape[1] - 1)])
            total_distance2 = np.mean([pmstir.distance(host_locations[2:, i - 1], host_locations[2:, i]) for i in range(1, host_locations.shape[1] - 1)])
            mean_step_size[s] = np.mean([total_distance1, total_distance2])

            # Visualize the trajectories
            # fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            # ax.plot(host_locations[0, :], host_locations[1, :], '-', label="host 1")
            # ax.plot(host_locations[2, :], host_locations[3, :], '-', label="host 2")

            # xlim = (-400, 400)
            # ylim = (-400, 400)

            # # Plot grid lines
            # for i in range(len(xlower)):
            #     ax.vlines(xlower[i], *ylim, color="black", linewidth=0.1, zorder=-10, alpha=0.25)
            #     ax.hlines(ylower[i], *xlim, color="black", linewidth=0.1, zorder=-10, alpha=0.25)

            # ax.set_ylim(*ylim)
            # ax.set_xlim(*xlim)
            # ax.set_xlabel("Easting (m)")
            # ax.set_ylabel("Northing (m)")
            # ax.set_title("Mean step size (m per hour) = {0:.2f}".format(mean_step_size[s]))
            # fig.savefig("../results/trajectory_{0}.pdf".format(drift), bbox_inches="tight")
            
            # Is host 2 infected?
            host2_inf = np.any(host_states[1, :] == 1)
            host2_end_state[s] = host2_inf
            # hosts_together_all_sims[s, :] = hosts_together

        # Get the key metrics
        print(np.mean(mean_step_size))
        all_step_sizes[d] = np.mean(mean_step_size)
        all_inf_risk[d] = np.mean(host2_end_state)

        if d == 0:
            # Get stationary distribution risk
            xvals = np.arange(minx, maxx + grid, step=grid)
            yvals = np.arange(minx, maxx + grid, step=grid)
            
            X, Y = np.meshgrid(xvals, yvals)
            xlong = X.ravel()
            ylong = Y.ravel()
            
            upper = np.vstack([xlong + grid*0.5, ylong + grid*0.5, xlong + grid*0.5, ylong + grid*0.5]).T
            lower = np.vstack([xlong - grid*0.5, ylong - grid*0.5, xlong - grid*0.5, ylong - grid*0.5]).T
            pbothinA = mvn_stationary.cdf(upper, lower_limit=lower).reshape(X.shape)

            p_t = np.sum(pbothinA)

            stationary_inf_risk[d] = 1 - (1 - (p_t*(1 - np.exp(-beta*deltat))))**(hours / deltat)  #1 - np.exp(-p_t*beta*hours)

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.semilogx(all_step_sizes, all_inf_risk , 'o', label="Simulated probability")
    ax.set_ylim(0, stationary_inf_risk[0]*3)
    xlim = ax.get_xlim()
    ax.hlines(stationary_inf_risk[0], *xlim, color="black", linestyle="--", label="Fast movement probability")
    ax.hlines(p_t*(1 - np.exp(-beta*hours)), *xlim, color="black", linestyle="-", label="Slow movement probability")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Average step size (meters per hour)")
    ax.set_ylabel("Probability of infection over 24 hours")
    ax.legend(loc="best")
    fig.savefig("../results/probability_of_infection_beta={0}.pdf".format(beta), bbox_inches="tight")
        