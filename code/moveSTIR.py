import numpy as np
import scipy.stats as stats
import pandas as pd
import scipy.interpolate as interp
from scipy.linalg import block_diag
# from patsy import dmatrix, bs, build_design_matrices
# from statsmodels.api import OLS
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel
import os

"""
Functions to implement MoveSTIR

"""

def transmission_kernel_summarize(h1_xy, h2_xy, pathogen_decay, distance_decay,
                                  dd_type="gaussian", with_params_dt=False,
                                  beta=1, lam=1):
    """
    Compute summaries of the directional kernel functions K_{1 <- 2} for 
    transmission weight for host 1 experiencing a force of
    infection from host 2 (host 1 running into host 2 trajectory). 

    This is substantially faster than trying to compute the entire transmission
    kernel. 
    
    Parameters
    ----------
    h1_xy : DataFrame
        Dataframe with columns time, x, and y. 
            'time': the time point where a spatial location was recorded. Must be equally spaced
            'x': The x-coordinate spatial location
            'y': The y-coordinate spatial location
        The acquiring host
    h2_xy : DataFrame
        Same as h1_xy. The ordered time columns must be the same for h1_xy and h2_xy
        The depositing host
    pathogen_decay : float
        Given exponential pathogen decay, the decay rate of the pathogen in the environment. Make sure the units match!
    distance_decay : float
        The distance decay parameter. Exact interpretation depends on dd_type.
            dd_type == "gaussian": Parameter is the mean distance of the half normal
            dd_type == "cutoff": Parameter is a the maximum distance beyond which transmission can't occur
    dd_type : str
        See options above
    with_params_dt : bool
        If True, multiply results by parameters beta and lambda and deltat. If
        False, don't.
    beta : float
        Acquisition rate. Default is 1
    lam : float
        Deposition rate. Default is 1
    
    Returns
    -------
    : deltat, [fois_direct, fois_indirect], None
        - Time step delta t
        - Marginal summaries of the transmission kernel
            - Direct foi felt by 1 from 2 (defined as the diagonal of the transmission kernel)
            - Indirect foi felt by 1 from  2
        - None

    Notes
    -----
    This function never stores the full K_{1 <- 2} transmission kernel in memory.

    NOTE: To be consistent with transmission_kernel, estimates are not yet
    multiplied by deltat.
    """
    
    host1_xy = h1_xy.sort_values(by="time").reset_index()
    host2_xy = h2_xy.sort_values(by="time").reset_index()
    assert np.all(host1_xy.time == host2_xy.time), "Ordered times are not equal for h1_xy and h2_xy"
    
    # Check on the size of the dataframes
    time = host1_xy.time.values
    deltat = time[1] - time[0]
    t = len(time)
    fois_direct = np.empty(t)
    fois_indirect = np.empty(t)
    fois_all = np.empty(t)

    # Loop over rows...will be slow but won't use much memory
    # Could potentially do this in chunks as well to effeciently vectorize
    for i in range(t):

        x1 = host1_xy.x.values[i]
        X2 = host2_xy.x.values

        y1 = host1_xy.y.values[i]
        Y2 = host2_xy.y.values

        dvect = np.sqrt(((x1 - X2)**2) + ((y1 - Y2)**2))

        # Weight distance
        if dd_type == "cutoff":
            # Top-hat contact function
            dist_weight =  tophat_cf(dvect, distance_decay)
        elif dd_type == "gaussian":
            # Gaussian contact function
            dist_weight = gaussian_cf(dvect, distance_decay)
        else:
            raise(TypeError("I don't recognize {0} for dd_type. Use 'cutoff' or 'gaussian'".format(dd_type)))

        # TODO: Allow for generic survival function
        x1_time = time[i] 
        time_diff = (x1_time - time)
        path_surv = np.exp(-pathogen_decay*time_diff)
        path_surv = np.where(time_diff >= 0, path_surv, 0) 

        if not with_params_dt:
            all_foi = path_surv * dist_weight
        else:
            all_foi = (path_surv * dist_weight)*beta*lam*deltat

        fois_all[i] = np.sum(all_foi)
        fois_direct[i] = all_foi[i] # Extract the diagonal
        fois_indirect[i] = fois_all[i] - fois_direct[i] # compute 

    return((deltat, (fois_direct, fois_indirect), None))


def transmission_kernel(h1_xy, h2_xy, pathogen_decay, distance_decay,
                        dd_type="gaussian", max_size=20000, file_path="Kmem.dat"):
    """
    Compute the directional kernel functions K_{1 <- 2} for transmission weight for host 1 experiencing a force of
    infection from host 2 (host 1 running into host 2 trajectory).
    
    Parameters
    ----------
    h1_xy : DataFrame
        Dataframe with columns time, x, and y. 
            'time': the time point where a spatial location was recorded. Must be equally spaced
            'x': The x-coordinate spatial location
            'y': The y-coordinate spatial location
        The acquiring host
    h2_xy : DataFrame
        Same as h1_xy. The ordered time columns must be the same for h1_xy and h2_xy
        The depositing host
    pathogen_decay : float
        Given exponential pathogen decay, the decay rate of the pathogen in the environment. Make sure the units match!
    distance_decay : float
        The distance decay parameter. Exact interpretation depends on dd_type.
            dd_type == "gaussian": Parameter is the mean distance of the half normal
            dd_type == "cutoff": Parameter is a the maximum distance beyond which transmission can't occur
    dd_type : str
        See options above
    max_size : int
        If h1_xy.shape[0] is greater than max_size, we will use memory mapping
        to compute transmission kernel and save the kernel to disk with name 
        file_path.  Otherwise, the kernel cannot be held in memory.
    file_path : str
        The full path where the transmission kernel will be saved. Note that
        these files can get huge very quickly. 
    
    Returns
    -------
    : deltat, K_{1 <- 2}, Dmat
        Time step delta t
        The directional transmission kernel functions approximated as a matrix
            - Rows are the acquiring host
            - Columns are the depositing host
        The distance matrix

    """
    
    host1_xy = h1_xy.sort_values(by="time").reset_index()
    host2_xy = h2_xy.sort_values(by="time").reset_index()
    assert np.all(host1_xy.time == host2_xy.time), "Ordered times are not equal for h1_xy and h2_xy"
    
    # Check on the size of the dataframes
    time = host1_xy.time.values
    deltat = time[1] - time[0]
    t = len(time)

    if host1_xy.shape[0] < max_size:

        lower_tri = np.tri(t)
        
        # X locations
        X1 = np.repeat(host1_xy.x.values, t).reshape((t, t)) * lower_tri 
        X2 = np.tile(host2_xy.x.values, t).reshape((t, t))

        # Y locations
        Y1 = np.repeat(host1_xy.y.values, t).reshape((t, t)) * lower_tri 
        Y2 = np.tile(host2_xy.y.values, t).reshape((t, t))

        # Distance matrix based on Euclidean distance
        dmat = np.sqrt(((X1 - X2)**2) + ((Y1 - Y2)**2)) * lower_tri

        # Weight distance
        if dd_type == "cutoff":
            # Top-hat contact function
            dist_weight =  tophat_cf(dmat, distance_decay) * lower_tri
        elif dd_type == "gaussian":
            # Gaussian contact function
            dist_weight = gaussian_cf(dmat, distance_decay) * lower_tri
        else:
            raise(TypeError("I don't recognize {0} for dd_type. Use 'cutoff' or 'gaussian'".format(dd_type)))
            
        # TODO: Allow for generic survival function
        X1_time = np.repeat(time, t).reshape((t, t))
        time_diff = (X1_time - time)
        path_surv = np.exp(-pathogen_decay*time_diff) * lower_tri

        K = (path_surv * dist_weight)

        return((deltat, K, dmat))

    else:

        # Matrix is too large to store in memory, use memory mapping
        Kmem = np.memmap(file_path, dtype=np.float32, mode="w+", shape=(t, t))

        # Loop over rows...will be slow but won't use much memory
        # Could potentially do this in chunks as well to effeciently vectorize
        for i in range(t):

            x1 = host1_xy.x.values[i]
            X2 = host2_xy.x.values

            y1 = host1_xy.y.values[i]
            Y2 = host2_xy.y.values

            dvect = np.sqrt(((x1 - X2)**2) + ((y1 - Y2)**2))

            # Weight distance
            if dd_type == "cutoff":
                # Top-hat contact function
                dist_weight =  tophat_cf(dvect, distance_decay)
            elif dd_type == "gaussian":
                # Gaussian contact function
                dist_weight = gaussian_cf(dvect, distance_decay)
            else:
                raise(TypeError("I don't recognize {0} for dd_type. Use 'cutoff' or 'gaussian'".format(dd_type)))

            # TODO: Allow for generic survival function
            x1_time = time[i] 
            time_diff = (x1_time - time)
            path_surv = np.exp(-pathogen_decay*time_diff)
            path_surv = np.where(time_diff >= 0, path_surv, 0) 

            Kmem[i, :] = (path_surv * dist_weight)

        # Flush memory
        del Kmem

        return((deltat, file_path, None))

def tophat_cf(d, distance_decay):
    """ 
    Top-hat contact function

    Parameters
    ----------
    d : array-like
        distance (or distance matrix)
    distance_decay : float
        The distance beyond which contact is 0

    Returns
    -------
    : float
        Probability density of contact
    """

    return((1 / (np.pi*distance_decay**2))*(d < distance_decay).astype(np.int))

def gaussian_cf(d, distance_decay):
    """
    Gaussian contact function

    Parameters
    ----------
    d : array-like
        distance (or distance matrix)
    distance_decay : float
        Mean distance of the half normal Gaussian distribution

    Returns
    -------
    : float
        Probability density of contact
    """
    return((1 / (4*distance_decay**2)) * 
            np.exp(-((np.pi*d**2) / (4*distance_decay**2))))



def set_discretized_values(min_size, max_size, bins):
    """
    Calculates the necessary parameters to use the midpoint rule to evalulate the discretized trajectory

    Parameters
    ----------
    min_size : lower time
    max_size : upper time
    bins : The number of bins in the discretized matrix

    Returns
    -------
    dict: min_size, max_size, bins, bnd (edges of discretized kernel), y (midpoints),
    h (width of cells)
    """

    # Set the edges of the discretized kernel
    bnd = min_size + np.arange(bins + 1)*(max_size-min_size) / bins

    # Set the midpoints of the discretizing kernel. Using midpoint rule for evaluation
    y = 0.5 * (bnd[:bins] + bnd[1:(bins + 1)])

    # Width of cells
    h = y[2] - y[1]

    return(dict(min_size=min_size,
                max_size=max_size,
                bins=bins, bnd=bnd, y=y,
                h=h))

def map_to_grid(xvals, yvals, xbounds, ybounds):
    """
    Assign x, y coordinates to grid values
    
    Parameters
    ----------
    xvals : array-like
        Array of x locations
    yvals : array-like
        Array of y locations
    xbounds : array-like
        The boundaries of grid cells on the x axis
    ybounds : array-like
        The boundaries of grid cells on the y axis
    
    Return
    ------
    : Tuple
        (xcoord, ycoord).
        xcoord is the x bin of the point
        ycoord is the y bin of the point
    
    Note
    ----
    If xcoord or ycoord is -1, then point not within bounds.
    """
    
    # Minus one so that bins starts at 0
    ycoord = np.argmax(yvals[:, np.newaxis] < ybounds, axis=1) - 1
    xcoord = np.argmax(xvals[:, np.newaxis] < xbounds, axis=1) - 1
    return((xcoord, ycoord))

def spatial_risk(x, deltat, calc):
    """ 
    Groupby area to calculate spatial risk
    
    Parameters
    ----------
    x: Groupby dataframe
        Has column "foi"
    deltat: float
        Time-step
    calc: str 
        "cumulative": Total cumulative hazard in area
        "average": Average FOI in area
        "cumulative_w_time": Total cumulative hazard with a column for time
    
    Returns
    -------
    : Series
    """
    
    if calc == "cumulative":
        # Total cumulative hazard in an area. Save time as well.
        names = {'risk': x.foi.sum()*deltat}
        return(pd.Series(names, index=['risk']))
    elif calc == "average":
        # Average FOI in an area
        names = {'risk': (x.foi.sum()*deltat) / (len(x.foi)*deltat)}
        return(pd.Series(names, index=['risk']))
    elif calc == "cumulative_w_time":
        names = {'risk': x.foi.sum()*deltat,
                 'time_in_area': len(x.foi)*deltat}
        return(pd.Series(names, index=['risk', 'time_in_area']))
    else:
        raise TypeError("{0} not recognized. Try 'cumulative' or 'average'".format(calc))
    

def calculate_spatial_risk(host_df, foi, deltat, xbounds, ybounds, calc="cumulative"):
    """
    Calculate the spatial risk from host's trajectory
    
    Parameters
    ----------
    host_df : Dataframe
        Ordered host movement locations with columns 'x' and 'y'
    foi : array-like
        The foi experienced by the host trajectory in host_df
    deltat : float
        Time-step
    xbounds : array-like
        The grid boundaries in the x dimension
    ybounds : array-like
        The grid boundaries in the y dimension
    
    Returns
    -------
    : Tuple
        spatial_foi : A DataFrame with columns 
                      'x': x grid location
                      'y': y grid location
                      'risk': Spatial transmission risk in grid
        grids : Dataframe
            map_to_grid result in dataframe form
    
    """
    
    # Continuous trajectory to grid cells
    xcoord, ycoord = map_to_grid(host_df.x.values, host_df.y.values, 
                                 xbounds, ybounds)
    
    # Plot grid 
    #     XX, YY = np.meshgrid(xbounds['bnd'], ybounds['bnd'])
    #     plt.plot(XX, YY, 'o', ms=1, color='red')
    #     plt.plot(host1.x.values, host1.y.values)
    #     plt.plot(host2.x.values, host2.y.values)
    #     plt.plot(xbounds['y'][xcoord], ybounds['y'][ycoord], 'o', ms=4)

    # The trajectory in grid form
    grids = pd.DataFrame([(x, y) for x, y in zip(xcoord, ycoord)], columns=["x", "y"])
    grids = grids.assign(foi=foi)

    spatial_foi = grids.groupby(['x', 'y']).apply(spatial_risk, deltat, calc).reset_index()
    
    return((spatial_foi, grids))


def align_trajectories(host1, host2):
    """
    Align host movement trajectories to the same time window

    Parameters
    ----------
    host1 : DataFrame
        Dataframe with columns time, x, and y. 
            'time': the time point where a spatial location was recorded. Must be equally spaced
            'x': The x-coordinate spatial location
            'y': The y-coordinate spatial location
    host2 : DataFrame
        Dataframe with columns time, x, and y. 
            'time': the time point where a spatial location was recorded. Must be equally spaced
            'x': The x-coordinate spatial location
            'y': The y-coordinate spatial location

    Returns
    -------
    : tuple
        (host1, host2), aligned (truncated) host trajectories
    """

     # Align time stamps. We are only comparing hosts where they overlap in time
    mintime = np.max([np.min(host1.time), np.min(host2.time)])
    maxtime = np.min([np.max(host1.time), np.max(host2.time)])
    host1 = host1[(host1.time >= mintime) & (host1.time <= maxtime)].reset_index(drop=False)
    host2 = host2[(host2.time >= mintime) & (host2.time <= maxtime)].reset_index(drop=False)
    return((host1, host2))


def block_diag_offset(values):
    """
    Make a special block diagonal matrix on lower, off diagonal

    Parameters
    ----------
    arrays : array-like
        List/array of square matrices

    Returns
    -------
    : matrix

    Example
    -------

    arrays = [1, 2, 3]

    R =[0 0 3
        1 0 0
        0 2 0]

    R is returned

    """

    val_final = values[-1]
    vals_other = values[:-1]

    n = 1
    p = len(values)
    n_full = n * p
    fullA = np.zeros((n_full, n_full))

    subA = np.diag(vals_other)
    fullA[:n, (n*p - n):(n*p)] = val_final
    fullA[n:, :(n*p - n)] = subA
    return(fullA)


def build_gp_kernel():
    """
    Kernel for Gaussian process
    """
    
    # Long-term trend in movement
    c1 = ConstantKernel()
    f1 = RBF(length_scale=20, length_scale_bounds=(1, 100))
    
    # Short-term trends in movement
    c2 = ConstantKernel()
    f2 = RBF(length_scale=0.1, length_scale_bounds=(1e-05, 1))

    # Quasi-periodicity
    c3 = ConstantKernel()
    f3 = ExpSineSquared(length_scale=1, periodicity=2, 
                        periodicity_bounds=(0.5, 100),
                        length_scale_bounds=(1e-5, 10))
    f4 = RBF(length_scale=1, length_scale_bounds=(1e-5, 5))
    
    wn = WhiteKernel(noise_level=0.0005, noise_level_bounds=(1e-08, 0.001))
    
    # Combination of kernels
    kernel = c1*f1 + c2*f2 + c3*f3*f4 #+ wn
    return(kernel)

def fit_gp_to_movement(time, xloc, yloc, interp_vals=None, step=None):
    """
    Fit a gaussian process to movement
    
    Parameters
    -----------
    time : array-like
        Time at which location was recorded. They do not need to be equally spaced
    xloc : array-like
        x-location at time 
    yloc : array-like
        y-location at time
    interp_vals : None or array
        If None, then the interpolated time is specified by the min and max of time and step
        If not None, interp_vals is an array with prespecified times to be interpolated. The range
        of interp_vals should be larger or equal to the range of time. The interpolated times are then
        selected as a subset or interp_vals
    step : None or float
        It interp_vals is None, the time step of the interpolation
    
    Returns
    -------
    : DataFrame of interpolated movements
        x, y, and time
    """
    
    if interp_vals is not None:
        time_pred = interp_vals[np.argmax(interp_vals >= np.min(time)):np.argmax(interp_vals >= np.max(time))]
    else:
        
        if step is not None:
            time_pred = np.arange(np.min(time), np.max(time), step=step)
        else:
            raise KeyError("Step value must be not None")
            
    interp_dat = np.empty((len(time_pred), 3))
    interp_dat[:, -1] = time_pred
    
    for i, loc_dat in enumerate([xloc, yloc]):
        
        loc_z = (loc_dat - loc_dat.mean()) / (loc_dat.std())
        gp_kernel = build_gp_kernel()

        print("Fitting GP...")
        gp_mod = GaussianProcessRegressor(kernel=gp_kernel, alpha=1e-10).fit(time[:, np.newaxis], loc_z)
        print("Complete")

        print("Predicting GP...")
        loc_mean = gp_mod.predict(time_pred[:, np.newaxis], return_cov=False)
        print("Complete")
        
        # Unstandarize
        interp_dat[:, i] = loc_mean*loc_dat.std() + loc_dat.mean()
    
    return(pd.DataFrame(interp_dat, columns=['x', 'y', 'time']))


def fit_interp_to_movement(time, xloc, yloc, interp_vals=None, step=None):
    """
    Fit a simple linear interpolator to movement trajectory. Interpolates x and y dimensions separately
    
    Parameters
    -----------
    time : array-like
        Time at which location was recorded. They do not need to be equally spaced
    xloc : array-like
        x-location at time 
    yloc : array-like
        y-location at time
    interp_vals : None or array
        If None, then the interpolated time is specified by the min and max of time and step
        If not None, interp_vals is an array with prespecified times to be interpolated. The range
        of interp_vals should be larger or equal to the range of time. The interpolated times are then
        selected as a subset or interp_vals
    step : None or float
        It interp_vals is None, the time step of the interpolation
    
    Returns
    -------
    : DataFrame of interpolated movements
        x, y, and time
    """

    
    if interp_vals is not None:
        time_pred = interp_vals[np.argmax(interp_vals >= np.min(time)):np.argmax(interp_vals >= np.max(time))]
    else:
        
        if step is not None:
            time_pred = np.arange(np.min(time), np.max(time), step=step)
        else:
            raise KeyError("Step value must be not None")
            
    interp_dat = np.empty((len(time_pred), 3))
    interp_dat[:, -1] = time_pred
    
    for i, loc_dat in enumerate([xloc, yloc]):
        
        loc_z = (loc_dat - loc_dat.mean()) / (loc_dat.std())
        interp_mod = interp.interp1d(time, loc_z)
        loc_mean = interp_mod(time_pred)
        
        # Unstandarize
        interp_dat[:, i] = loc_mean*loc_dat.std() + loc_dat.mean()
    
    return(pd.DataFrame(interp_dat, columns=['x', 'y', 'time']))


def fit_bspline_to_movement(time, xloc, yloc, interp_vals=None, step=None, df=None):
    """
    Fit a B-spline to movement
    
    Parameters
    -----------
    time : array-like
        Time at which location was recorded. They do not need to be equally spaced
    xloc : array-like
        x-location at time 
    yloc : array-like
        y-location at time
    interp_vals : None or array
        If None, then the interpolated time is specified by the min and max of time and step
        If not None, interp_vals is an array with prespecified times to be interpolated. The range
        of interp_vals should be larger or equal to the range of time. The interpolated times are then
        selected as a subset or interp_vals
    step : None or float
        It interp_vals is None, the time step of the interpolation
    df : int or None
        The degrees of freedom of the bpsline (parameters, essentially).
        If None, defaults to 1/2 the number of data points.
    
    
    Returns
    -------
    : DataFrame of interpolated movements
        x, y, and time
    """
    
    # Degrees of freedom for Bspline
    if df is None:
        df = np.int(len(time) / 2)
    
    # Set interpolation values
    if interp_vals is not None:
        time_pred = interp_vals[np.argmax(interp_vals >= np.min(time)):np.argmax(interp_vals >= np.max(time))]
    else:
        
        if step is not None:
            time_pred = np.arange(np.min(time), np.max(time), step=step)
        else:
            raise KeyError("Step value must be not None")
            
    interp_dat = np.empty((len(time_pred), 3))
    interp_dat[:, -1] = time_pred
    
    for i, loc_dat in enumerate([xloc, yloc]):
        
        loc_z = (loc_dat - loc_dat.mean()) / (loc_dat.std())
        
        XBspline = dmatrix("bs(x, df={0}, include_intercept=True)".format(df), 
                           {'x': time})
        fit_ols = OLS(loc_z, XBspline).fit()
        Xnew = build_design_matrices([XBspline.design_info], {'x': time_pred})
        loc_mean = fit_ols.predict(Xnew)
        
        # Unstandarize
        interp_dat[:, i] = loc_mean*loc_dat.std() + loc_dat.mean()
    
    return(pd.DataFrame(interp_dat, columns=['x', 'y', 'time']))

def movement_R0_from_avg_foi(F, gamma):
    """
    Given the avg maximum FOI matrix F, compute R0

    Parameters
    ----------
    F : array
        An n by n array with the average FOI individual interactions.
        The columns are depositing hosts and rows are acquiring hosts.
    gamma : float
        The loss of infection rate
    """

    n = F.shape[0]
    U = np.diag(np.repeat(-gamma, n))
    R = np.dot(F, np.linalg.inv(-U))
    R0 = np.max(np.abs(np.linalg.eigvals(R)))
    return((R, R0, F, U))

def movement_R0(host_trajs, params, perturb=None):
    """
    Calculate R0 from movement trajectories for an SIS model
    
    Parameters
    ----------
    host_trajs : dict
        Dictionary with key words 'h1', 'h2', etc. Each key word looks up a dataframe
        with observed host movement trajectories
    params : dict
        Contains parameters used to calculate R0. 
            'distance_decay', 'pathogen_decay', 'beta', 'lambda', 'gamma', and 
            'dd_type'. See 'transmission_kernel' for definitions of parameters.
            'gamma': Loss of infection rate
    perturb : None or dict
        If dict, contains the keywords
        'xbound', 'ybound', 'delta' as defined in spatial perturbation kernel
    
    Returns
    -------
    : tuple
        (transmission kernel R, R0)
    """
    
    # Unpack params
    distance_decay = params['distance_decay']
    pathogen_decay = params['pathogen_decay']
    β = params['beta']
    λ = params['lambda']
    dd_type = params['dd_type']
    γ = params['gamma']
    
    host_keys = np.sort(np.array(list(host_trajs.keys())))
    num = len(host_trajs[host_keys[0]])
    time = host_trajs[host_keys[0]].time.values
    deltat = time[1] - time[0]

    # Build the loss of infection kernel
    A = block_diag_offset(np.repeat(1 - γ*deltat, num))
    
    Z = np.zeros(A.shape)

    # Build the transmission kernels
    Ffull = [] # Fecundity matrix
    Ufull = [] # Survival matrix
    Jfull = [] # Jacobian
    
    # Acquire loop
    for h1 in host_keys:

        Fsub = []
        Usub = []
        Jsub = []
        
        # Deposit loop
        for h2 in host_keys:

            if h1 == h2:
                # Diagonal
                Usub.append(A)
                Fsub.append(Z)
                K = A
            else:
                # Transmission kernel
                K = transmission_kernel(host_trajs[h1], host_trajs[h2], pathogen_decay, 
                                        distance_decay, dd_type=dd_type)[1]
                K = (K * β * λ * deltat) # Force of infection
                Usub.append(Z)
                Fsub.append(K)

            Jsub.append(K)

        Ffull.append(Fsub)
        Ufull.append(Usub)
        Jfull.append(Jsub)

    # Check against J
    F = np.block(Ffull)
    U = np.block(Ufull)
    J = np.block(Jfull)
    
    if perturb is not None:
        perturb['host_trajs'] = host_trajs
        Pmat =  spatial_perturbation_kernel(**perturb)
        F = F * Pmat # Perturbation

    # Calculate R0
    I = np.eye(F.shape[0])
    # Need to multiply by delta to convert back to time units
    R = np.dot(F, np.linalg.inv((I - U)) * deltat)
    R0 = np.max(np.abs(np.linalg.eigvals(R)))
    
    return((R, R0, F, U))


def spatial_perturbation_kernel(host_trajs, xbound, ybound, delta):
    """
    Perturb all host kernel functions by delta when hosts are in grid cell 
    bounded by xbound and ybound. Perturbs aquisition -- when a host is in the
    location perturbs its ability to uptake from that location. 
    
    Parameters
    ----------
    host_trajs : dict
        Keywords are h1, h2, h3, etc. Each entry is the sorted movement Dataframe with
        columns time, x, and y
    xbound : array
        Length 2. Upper and lower bound on x coordinate of grid cell
    ybound : array
        Length 2. Upper and lower bound on coordinate of grid cell
    delta : float
        (1 - delta) is the perturbation
    
    Return
    ------
    : Pmat_full
        Array with dimensions (len(host_keys)*time, len(host_keys)*time) that can be element-wise
        multiplied by the kernel function to perturb the kernel function when hosts are in
        particular spatial locations

    Notes
    -----
    One could also envision perturbing deposition. However, because we assume
    that pathogens instantaneously spread out on the environment to occupy some
    circular area, perturbing deposition at the level of a grid cell will not
    necessarily yield the same answer as perturbing acquisition. 
    """

    Pmats_acquire = []
    # Pmats_despoit = []
    host_keys = np.sort(np.array(list(host_trajs.keys())))
    
    for h in host_keys:

        indx, indy = map_to_grid(host_trajs[h].x.values, host_trajs[h].y.values, 
                                 xbound, ybound)
        incell = np.bitwise_and(indx != -1, indy != -1)#.astype(np.int)
        
        cell_perturb = np.where(incell, 1 - delta, 1)

        # For a single depositing point, perturb host aquisition (DOWN rows)
        perturb_mat_acquire = (np.tri(len(incell))) * cell_perturb[:, np.newaxis]

        # For a single acquiring point, perturb host deposition (ACROSS rows)
        # perturb_mat_deposit = (np.tri(len(incell))) * cell_perturb

        Pmats_acquire.append(perturb_mat_acquire)
        # Pmats_despoit.append(perturb_mat_deposit)

    # Pmat_full_deposit = np.block([Pmats_despoit for i in range(len(host_keys))])
    Pmat_full_acquire = np.block([[Pmats_acquire[i] for j in range(len(host_keys))] 
                           for i in range(len(host_keys))])
    return(Pmat_full_acquire)


def simulate_sis_model(steps, host_trajs, params, 
                       initial_values=1e-8):
    """
    Simulate the individual-level SIS model

    Parameters
    ----------
    steps: int
        Each step corresponds to an entire simulation over the movement
        trajectories.
    host_trajs : dict
        Dictionary with key words 'h1', 'h2', etc. Each key word looks up a dataframe
        with observed host movement trajectories
    params : dict
        Contains parameters used to calculate R0. 
            'distance_decay', 'pathogen_decay', 'beta', 'lambda', and 'dd_type'. 
            See 'transmission_kernel' for definitions of parameters.
    initial_values : float or array-like
        Should be either a single value or have the same length as the number
        of hosts. Initializes the initial infection probability of all hosts or
        the first host.
    """

    # Unpack params
    distance_decay = params['distance_decay']
    pathogen_decay = params['pathogen_decay']
    β = params['beta']
    λ = params['lambda']
    dd_type = params['dd_type']
    γ = params['gamma']
    
    sim_steps = steps
    host_keys = list(host_trajs.keys())
    h1 = host_trajs[host_keys[0]]
    deltat = h1.time.values[1] - h1.time.values[0]
    len_of_season = len(h1) # Length of movement trajectory
    time_steps = len_of_season * sim_steps # Number of simulation steps
    num_hosts = len(host_keys)

    # Structure of rows host 1 S, host 1 I, host 2 S, host 2 I, host3 S, host 3 I, etc.
    sim_probs = np.zeros((num_hosts*2, time_steps))

    # Initialize either all hosts or the first host
    initial_values_arr = np.atleast_1d(initial_values)
    if len(initial_values_arr) > 1:
        sim_probs[::2, 0] = 1 - initial_values_arr
        sim_probs[1::2, 0] = initial_values_arr
    else:
        sim_probs[::2, 0] = 1
        sim_probs[1::2, 0] = 0
        sim_probs[0, 0] = 1 - initial_values_arr[0]
        sim_probs[1, 0] = initial_values_arr[0]

    # Build Kmats. Only need to compute these once
    Kmats = []
    for h1 in host_keys:
        
        tKs = []
        for h2 in host_keys:
            if h1 == h2:
                K = np.zeros((len_of_season, len_of_season))
            else:
                K = transmission_kernel(host_trajs[h1], host_trajs[h2], 
                                        pathogen_decay, 
                                        distance_decay,
                                        dd_type=dd_type)[1]
                K = K * β * λ * deltat
                
            tKs.append(K)
        Kmats.append(tKs)
            
    Kmats = np.block(Kmats)

    for t in range(time_steps - 1):
        
        # What iteration are we in?
        iteration = np.int64(np.floor(t / len_of_season))
        in_season_t = t % len_of_season
        
        # Make a vector of infected probabilities over time in the current season
        infected_probs = np.ravel(sim_probs[1::2, iteration*len_of_season:((iteration + 1)*len_of_season)])
        
        # trans_prob
        fois = (infected_probs * Kmats).sum(axis=1)
        trans_probs = np.reshape(1 - np.exp(-fois*deltat), (num_hosts, len_of_season))
        recovery_prob = 1 - np.exp(-γ*deltat)
        
        current_probs = sim_probs[:, t]
        
        # Build transition matrix
        Tmats = []
        for i in range(num_hosts):
            
            host_mat = np.array([[1 - trans_probs[i, in_season_t], recovery_prob],
                                 [trans_probs[i, in_season_t], 1 - recovery_prob]])
            Tmats.append(host_mat)
            
        Tmats = block_diag(*Tmats)
        
        # Project foward one step
        sim_probs[:, t + 1] = np.dot(Tmats, current_probs)

    return(sim_probs)


### Functional Movement Models ######


def build_H(nvals, m, lower, upper, phi_H):
    """
    Gaussian basis function for functional movement model 

    (Hooten and Johnson 2017, JASA)

    Parameters
    ----------
     nvals : array-like
        Array of points at which to simulate the model
    m : int
        Number of equally spaced time-steps used to approximate the integral
    lower : float
        The lower bound of the time interval
    upper : float
        The upper bound of the time interval

    Returns
    -------
    : n by m Matrix
        The discretized Gaussian basis function that defines the functional
        movement model

    """
    mvals = np.linspace(lower, upper, num=m)
    n = len(nvals)
    
    nvals = np.atleast_1d(nvals).ravel()

    H = np.empty((n, m))
    for n, t in enumerate(nvals):
        norm = stats.norm(loc=t, scale=phi_H)
        norm_constant = norm.sf(lower) - norm.sf(upper)
        H[n, :] = norm.sf(mvals) #(norm.sf(mvals) - norm.sf(tupper)) / norm_constant
    
    return(H)


def simulate_movement(nvals, m, sigma_process, sigma_measure, phi_H, 
                      init_position=np.array([0, 0]), 
                      lower=0, upper=1):
    """
    Parameters
    ----------
    nvals : array-like
        Array of time points at which to simulate the model. In this function,
        They are assumed to be equally spaced.
    m : int
        Number of equally spaced time-steps used to approximate the integral
    sigma_process : float
        The standard deviation of the white noise process. Implictly includes sqrt(Δt)
    sigma_measure : float
        The standard deviation of the measurement error processs.
    phi_H : float
        The shape parameter of the convolution kernel
    lower : int
        Lower bound of the time interval
    upper : int
        Upper bound of the time interval
    init_position : int
        Where the movement starts

    Returns
    -------
    : DataFrame
        Simulated movement at specific with x and y locations
        Columns in the DataFrame are
        'time': n values spaced between lower and upper times
        'x': x location
        'y': y location
    
    """
    
    μ0 = init_position[:, np.newaxis]
    I = np.eye(2)
    n = len(nvals)
    
    # White noise
    e = stats.norm(loc=0, scale=sigma_process).rvs(2*m)[:, np.newaxis]
    
    # Gaussian convolution
    H = build_H(nvals, m, lower, upper, phi_H)
    ones = np.ones(n)[:, np.newaxis]
    μ = np.kron(μ0, ones) + np.dot(np.kron(I, H), e)
    
    # Measurement error
    s = stats.norm(loc=μ.ravel(), scale=sigma_measure).rvs(size=μ.shape[0])
    x = s[:n]
    y = s[n:]
    
    time = np.linspace(lower, upper, num=n)
    sim = pd.DataFrame(dict(time=time, x=x, y=y))
    return(sim)

