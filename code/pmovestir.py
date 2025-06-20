from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import scipy.stats as stats
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RectBivariateSpline
import moveSTIR as mstir

def calculate_ud_on_grid(host_locs, bounds, step):
    """
    Calculate the kernel density estimate given host locations. Note this is not accounting
    for autocorrelation in movement. 
    
    Parameters
    ----------
    host_locs : array-like
        2 x n array of x, y locations
    bounds : tuple
        (xmin, xmax, ymin, ymax)
    step : int
        Number of grid cells per x and y
    
    Returns
    -------
    : dict
        "Z": Predicted density
        "kde": The kernel density function
    """

    xmin, xmax, ymin, ymax = bounds
    host_bounds = (np.min(host_locs[0, :]), np.max(host_locs[0, :]),
                   np.min(host_locs[1, :]), np.max(host_locs[1, :]))
    X, Y = np.meshgrid(np.arange(xmin - step*10, xmax + step*10, step=step), 
                       np.arange(ymin - step*10, ymax + step*10, step=step))
    positions = np.vstack([X.ravel(), Y.ravel()])
    kde = gaussian_kde(host_locs)
    Z = np.reshape(kde(positions).T, X.shape)
    return({"XYZ": (X, Y, Z), "kde": kde, "host_bounds": host_bounds})

def hdi(k, Z, percent):
    """
    Get the highest density interval for a matrix
    
    Parameters
    ----------
    k : float
        Value that all cells are above. This parameter is optimized
    Z : array-like
        2D array of PROBABILITIES (make sure you apply appropriate transformation)
    percent : float
        The density contained in the interval
    
    """
    return(np.sum(Z[Z > k]) - percent)

def get_overlap_polygon(host1_UD, host2_UD, X, Y, percent=0.75, lower_bound=1e-10, 
                        starting_value=None, color="black"):
    """
    Find the polyon that specifies the X% contact area of overlap
    
    Parameters
    ----------
    host1_UD, host2_UD: array-like
        2D matricies of the kernel densities on a grid
    X, Y : array-like
        The grid points where the host UDs are calculated
    percent : float
        The percent of the contact distribution that the contact interval should contain
    lower_bound : float
        When specifying the lower bound for the contour, you can't chose exactly zero 
        (where zero specifies everything below the the percent contour)
    
    
    Returns
    -------
    : Shapely polygon
        The boundary of the area with X% of the contact overlap probability density
        
    """
    
    Zcontact = host1_UD * host2_UD
    Zcontact = Zcontact / np.sum(Zcontact)
    
    if starting_value is None:
        starting_value = np.mean(Zcontact)
        
    kbest = fsolve(hdi, [starting_value], args=(Zcontact, percent))
    Zcontact[Zcontact < kbest] = 0 # Mask everything else
    
    # Get contour...probably a more efficient way to do this
    cs = plt.contour(X, Y, Zcontact, levels=[lower_bound], colors=color)
    verts = cs.collections[0].get_paths()[0].vertices
    poly = Polygon(verts)
    return(poly)


def random_points_in_polygon(polygon, number):
    """
    Slow approach for drawing random point in polyo
    
    Parameters
    ----------
    polygon : shapely polygon
        The area in which to draw random points
    number : int
        Number of random points to draw.
    
    Returns
    -------
    : list of Shapely points
    """
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        
        if polygon.contains(pnt):
            points.append(pnt)
    return(points)

def make_correlation_map_in_area(area, number, host1_locs, host2_locs, contact_dist, seed=10,
                                 cap_style=1):
    """
    Generate a spatial correlation map of host co-occurrence in an area of interest
    
    Parameters
    ----------
    area : Shapely polygon
        The area over which to draw random points to estimate correlation
    number : int
        Number of random points to draw in area
    host1_locs, host2_locs : array
        2D array with 2 rows and n columns. Gives the spatial locations of hosts
    contact_dist : float
        The maximum distance between two hosts that would constitute a potential contact
    cap_style : int
        1: A circular buffer with radius contact_dist / 2
        3: A square buffer with side length = contact_dist
    
    Returns
    -------
    : GeoPandas Dataframe
        For each random spatial point, whether or not host 1 and host 2 points are in that
        random buffered area.
    
    
    Notes
    -----
    What is happening here is that we are calculating the correlation of co-occurrence
    for two individuals in an area of use.  Co-occurrence occurs when hosts are within some
    minimum distance of each other.  
    
    We are not currently accounting for temporal autocorrelation between points, 
    which we could potentially do with a large regression and an AR1 regression term.
    """
    
    np.random.seed(seed)
    pts = random_points_in_polygon(area, number)

    dfhost1 = gpd.GeoDataFrame({'geometry': [Point(vect) for vect in host1_locs.T]})
    dfhost2 = gpd.GeoDataFrame({'geometry': [Point(vect) for vect in host2_locs.T]})

    slopes = []
    total_use = []
    all_dfs = [] 
    for p, pt in enumerate(pts):

        # Make polygon. Divide by 2 because we are buffering around a random point.
        # Therefore, hosts within this circle will be at max contact_dist away from
        # each other.
        
        tarea = pt.buffer(contact_dist / 2, cap_style=cap_style)

        dfpoly = gpd.GeoDataFrame({'geometry': [tarea]})

        # Get points in poly host 1
        host1_in = gpd.tools.sjoin(dfhost1, dfpoly, predicate="within", how='left')
        host1_in.loc[host1_in.index_right == 0, "index_right"] = 1
        host1_in.loc[host1_in.index_right.isna(), "index_right"] = 0

        # Get points in poly host2
        host2_in = gpd.tools.sjoin(dfhost2, dfpoly, predicate="within", how='left')
        host2_in.loc[host2_in.index_right == 0, "index_right"] = 1
        host2_in.loc[host2_in.index_right.isna(), "index_right"] = 0
        
        tdf = pd.DataFrame({'host1_in': host1_in.index_right.values,
                            'host2_in': host2_in.index_right.values,
                            'pt_id': p})
        all_dfs.append(tdf)
    
    pt_gdf = pd.concat(all_dfs)
    return((pt_gdf, pts))



def build_correlation_surface(pts_gdf, area, grid_num=100, value_name="cor"):
    """
    Build the correlation surface from the spatial correlation points
    
    Parameters
    -----------
    pts_gdf : GeoPandas DataFrame
        Contains columns 
            'cor': Correlation estimate for point in space
            'geometry': The Shapely object for a point in space
    area : Shapely polygon
        The polygon of contact overlap
        
    Return
    ------
    : tuple
        (X, Y, Z) -> 
    """
    
    x, y = list(zip(*[list(pt.coords)[0] for pt in pts_gdf.geometry.values]))
    z = pts_gdf.loc[:, value_name].values
    xy = np.array(list(zip(x, y)))
    
    # Using this interpolator for now because it returns nans outside the boundaries.
    # Bspline interpolator probably a better option.
    func = CloughTocher2DInterpolator(xy, z)
    #LinearNDInterpolator(xy, z, fill_value=0)
    #CloughTocher2DInterpolator(xy, z)

    minx, miny, maxx, maxy = area.bounds
    X, Y = np.meshgrid(np.linspace(minx, maxx, grid_num), np.linspace(miny, maxy, grid_num))
    grid_pts = gpd.GeoDataFrame({'geometry': [Point(x, y) for x, y in zip(X.ravel(), Y.ravel())]})
    area_df = gpd.GeoDataFrame({'geometry': [area]})
    points_in = gpd.tools.sjoin(grid_pts, area_df, predicate="within", how='left')
    ind = points_in.index_right == 0

    # Plot the correlation surface
    tX = X.ravel()[ind]
    tY = Y.ravel()[ind]
    tZ = func(list(zip(tX, tY)))
    #tZ = func(tX, tY, grid=False)
    
    # Build and retur
#     def cor_func(x, y):
#         if(x < minx or x > maxx or y < miny or y > maxy):
#             res = 0
#         else:
#             res = func(x, y).ravel()[0]
#         return(res)
    cor_func = func
    
    return(((tX, tY, tZ), cor_func))

def integrate_box_from_pt(pt, ud, contact_distance):
    """
    Integrate a UD within a box with a center at pt and sides of 
    length contact_distance
    
    Parameters
    -----------
    pt : Shapely pt
    ud : kde 
    contact_distance : float
        Contact distance
    
    Return
    ------
    : Probability integrated at the box
    """
    X = np.concatenate(pt.xy)[0]
    Y = np.concatenate(pt.xy)[1]
    Xlower = X - contact_distance / 2
    Xupper = X + contact_distance / 2
    Ylower = Y - contact_distance / 2
    Yupper = Y + contact_distance / 2
    lower = (Xlower, Ylower)
    upper = (Xupper, Yupper)
    
    return(ud.integrate_box(lower, upper))

def distance(p1, p2):
    return(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


def interpolate_all_trajectories(step, dat):
    """
    Linearly interpolate all of the deer trajectories to the step scale.

    Parameters
    ----------
    step : int
        Step size on which to linear interpolate pig trajectories
    dat : DataFrame
        All pig data 

    Returns
    -------
    : dict
        Keys are host IDs, values are interpolated movement trajectories
        to the "step" scale.  
    """

    unq_collar = np.unique(dat.individual_ID)

    # Ensure all pigs are aligned when interpolating
    interp_vals = np.arange(dat.unix_time.min(), dat.unix_time.max() + step, step=step)

    all_fitted = {}
    for unq_ind in unq_collar:

        trial_dat = dat[dat.individual_ID == unq_ind]

        # Remove any of the same datetimes
        trial_dat = trial_dat[~trial_dat.datetime.duplicated()].sort_values("datetime").reset_index(drop=True)

        min_time = trial_dat.unix_time.min()
        time = trial_dat.unix_time.values
        xloc = trial_dat.UTMx.values
        yloc = trial_dat.UTMy.values

        fitted = mstir.fit_interp_to_movement(time, xloc, yloc, interp_vals=interp_vals)
        all_fitted[unq_ind] = fitted

    return(all_fitted)

def randomize_day(host):
    """
    Randomize locations by day following Speigel et al. 2016

    Parameters
    ----------
    host : DataFrame
        Has columns time
    """
    
    host = host.assign(datetime=lambda x: pd.to_datetime(x.time*60*10**9))
    host = host.assign(month_day = lambda x: x.datetime.dt.month.astype(str) + "_" + x.datetime.dt.day.astype(str))

    unique_days = host.month_day.unique()
    unique_days_rand = unique_days.copy()
    np.random.shuffle(unique_days_rand)
    day_map = pd.DataFrame({'month_day': unique_days, 'month_day_rand': unique_days_rand})
    host = (host.set_index("month_day")
          .join(day_map.set_index("month_day"))
          .reset_index()
          .assign(rand_datetime=lambda x: [pd.datetime(year, month, day, hour, minute, second)
                                              for year, month, day, hour, minute, second,
                                             in zip(x.datetime.dt.year, 
                                              [int(x[0]) for x in x.month_day_rand.str.split("_")],
                                              [int(x[1]) for x in x.month_day_rand.str.split("_")],
                                               x.datetime.dt.hour, 
                                               x.datetime.dt.minute,
                                               x.datetime.dt.second)])
          .sort_values("rand_datetime")[['x', 'y', 'rand_datetime']]
          .assign(time= lambda x: x.rand_datetime.astype(np.int64) / (60*10**9))[['x', 'y', 'time']])
    return(host)


def inv_minmax(x, lower=0, upper=1, tol=1e-10):
    """
    Inverse of minmax
    """
    trans = (np.exp(x)*upper + lower) / (1 + np.exp(x))
    trans[trans == lower] = lower + tol
    trans[trans == upper] = upper - tol
    return(trans)


def minmax(x, lower=0, upper=1):
    return(np.log((x - lower) / (upper - x)))



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


# Going to run into memory issues with this approach!
def matrix_correlation(host1, host2, flatX, flatY):
    """
    Compute the correlation surface between host1 and host2 occurences at all
    locations given by the flatX and flatY pairs

    Parameters
    ----------
    host1 : DataFrame 
        With columns x and y that indicate locations
    host2 : DataFrame
        With columns x and y that indicate locations
    flatX : array-like
        All x points to evaluation correlation
    flatY : array-like
        
    """
    
    host1_xlocs = host1.x.values
    host1_ylocs = host1.y.values

    host2_xlocs = host2.x.values
    host2_ylocs = host2.y.values
    
    flatX_lower, flatX_upper = flatX
    flatY_lower, flatY_upper = flatY
    
    # Check host 1 and host 2 in cells
    inX1 = np.bitwise_and((flatX_lower < host1_xlocs[:, np.newaxis]), (flatX_upper >= host1_xlocs[:, np.newaxis]))
    inY1 = np.bitwise_and((flatY_lower < host1_ylocs[:, np.newaxis]), (flatY_upper >= host1_ylocs[:, np.newaxis]))
    incell1 = (inX1 * inY1).astype(np.int64)

    inX2 = np.bitwise_and((flatX_lower < host2_xlocs[:, np.newaxis]), (flatX_upper >= host2_xlocs[:, np.newaxis]))
    inY2 = np.bitwise_and((flatY_lower < host2_ylocs[:, np.newaxis]), (flatY_upper >= host2_ylocs[:, np.newaxis]))
    incell2 = (inX2 * inY2).astype(np.int64)
    
    # Compute the vectorized correlation
    sd1 = np.std(incell1, axis=0)
    sd2 = np.std(incell2, axis=0)
    mean1 = np.mean(incell1, axis=0)
    mean2 = np.mean(incell2, axis=0)
    mean12 = np.mean(incell1 * incell2, axis=0)
    cor12 = (mean12 - mean1*mean2) / (sd1 * sd2)
    #cor12[np.isnan(cor12)] = 0
    return(cor12)
