import numpy as np
from shapely import geometry
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import os
import itertools
import scipy.stats as stats
import scipy.spatial as spatial
import multiprocessing as mp
import glob
import logging
from scipy.optimize import fsolve
from scipy.interpolate import NearestNDInterpolator
import rasterio
import rasterio.plot
from numba import njit, prange
from scipy.linalg import logm

logging.basicConfig(filename='build_landscape.log', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.info("Initializing logger")

"""
Functions and classes for landscape-scale simulations of epi risk
"""

class Individual(object):
    """
    An individual deer that has a centroid, individual ID, group ID, and a UD
    """

    def __init__(self, pt, individual_id,  group_id, sex=None, age=None, dispersed=False):
        """
        Parameters
        ----------
        pt : Shapely Point object
            Gives the "centered" location on the landscape.  Note this is not necessarily the center of use.
        individual_id : int
            Unique individual id
        group_id : int
            The group to which the individual belongs
        sex : string
            "M" for male of "F" for female
        age : string
            "Y" for yearling, "F" for fawn, and "A" for adult
        dispersed : bool
            If True, the individual has already dispersed. If False, hasn't dispersed.
        """

        self.pt = pt
        self.individual_id = individual_id
        self.group_id = group_id
        self.sex = sex
        self.age = age
        self.mom_id = np.nan
        self.alive = True
        self.death_time = np.nan
        self.birth_time = np.nan
        self.dispersed = dispersed

    def plot_ud(self, contour=False, ax=None):
        """
        Plot the individual UD

        Parameters
        ----------
        contour : bool

        """

        if hasattr(self, "xvals") and hasattr(self, "yvals") and hasattr(self, "ud_path"):

            if ax is None:
                fig, ax = plt.subplots(1, 1)

            Z = self.get_ud()
            X, Y = np.meshgrid(self.xvals, self.yvals)

            if not contour:
                ax.pcolormesh(X, Y, Z, shading="auto", zorder=0)
            else:
                ax.contour(X, Y, Z, zorder=0)
        else:
            print("No UD has been assigned to individual {0}".format(self.individual_id))

    def get_ud_hdi(self, interval=0.5, starting_value=1e-10):
        """
        Get the HDI of a UD at some given interval

        Parameters
        ----------
        interval : float
            From 0 to 1, Percent of the HPD
        starting_value : float
            Initial value
        """

        if hasattr(self, "Z"):
            Z = self.Z
        elif hasattr(self, 'ud_path'):
            Z = pd.read_pickle(self.ud_path)
        else:
            print("Indiviual does not have a UD")

        X, Y = np.meshgrid(self.xvals, self.yvals)
        poly = get_hdi_polygon(Z, X, Y, starting_value=1e-10, percent=0.50)
        self.hdi_poly = (poly, interval)
        return(self.hdi_poly)

    def get_ud(self):
        """
        Return the UD for an individual
        """

        if hasattr(self, "Z"):
            Z = self.Z
        elif hasattr(self, "ud_path"):
            Z = pd.read_pickle(self.ud_path)
        else:
            raise AttributeError("Indiviual {0} does not have UD".format(self.individual_id))

        if hasattr(self, "Zmask"):
            return(Z * self.Zmask)
        else:
            return(Z)

    def delete_mask(self):
        """
        If masking exists, delete mask
        """
        delattr(self, "Zmask")

    def set_birth_time(self, birth_time):
        self.birth_time = birth_time

    def set_dispersed(self, dispersed):
        self.dispersed = dispersed


class Landscape(object):
    """
    Landscape object that holds geographic and biological details of the landscape
    """

    def __init__(self, bounds, grid_size, buffer, crs=None):
        """

        Expects the units of the landscape to be in meters

        Parameters
        ----------
        bounds : tuple
            (xlower, ylower, xupper, yupper)
        grid_size : float
            The grid size in meters
        buffer : float
            The buffer around the landscape. Needed because individuals can 
            have UDs that extend beyond the landscape borders depsite their
            "center" being on the landscape.

        Notes
        -----
        Expects the units of the landscape to be in meters
        """

        xlower, ylower, xupper, yupper = bounds
        coords = [(xlower, ylower), (xlower, yupper), (xupper, yupper), (xupper, ylower)]
        self.polygon = geometry.Polygon(coords)
        self.polygon_gpd = gpd.GeoDataFrame(geometry=[self.polygon])
        self.area = self.polygon.area
        self.buffer = buffer
        self.grid_size = grid_size
        self.bounds = bounds

        # Grid cells
        self.xvals = np.arange(xlower - buffer, xupper + grid_size + buffer, 
                               step=grid_size)
        self.yvals = np.arange(ylower - buffer, yupper + grid_size + buffer, 
                               step=grid_size)

        # Get grid cells within the non-buffered boundary
        self.xvals_bounds = self.xvals[np.bitwise_and(self.xvals >= xlower, self.xvals <= xupper)]
        self.yvals_bounds = self.yvals[np.bitwise_and(self.yvals >= ylower, self.yvals <= yupper)]

        # Hold any rasters that are passed in at a later stage
        self.landscape_rasters = {}

    def assign_raster(self, src, name, band):
        """
        Convert a raster source to the landscape and save as a numpy array.

        NOTE: Assumes that your raster is already in the same CRS as the
        landscape

        TODO: Going to need to do this differently because we are going to 
        run out of RAM really quickly doing it this way.  Probably want to
        save this as a Raster to disk and access from disk.  Let's do it this
        way for now.

        Parameters
        ----------
        src : raster source
            The
        name : string
            The name of your raster for storage
        band : int
            Specify the band of your raster
        """

        # Note: can save this to a memmap array or raster and query from memory
        # This will be more efficient and will keep us from running into memory
        # issues.
        Z = interpolate_raster_to_landscape(self, src, band=band)
        self.landscape_rasters[name] = Z


    def mask_uds_with_raster(self, name, condition):
        """
        For the raster given in name, mask the UDs of every individual based on
        condition.  This allows us to knock out parts of the landscape and
        see how it affects disease spread. Note that the knockout is assuming
        that shedding or suscpetibility is being altered because we are
        not redistributing the UD among remaining cells. 

        Parameters
        ----------
        name : string
            Name of raster in self.landscape_rasters
        condition : string
            Some boolean condition on which to evaluate the raster. 
            For example, "< 20".

        Note
        ----
        This masking permenantly overwrites the UDs in ram and on disk. There
        might be a better way to do this, but this avoids having multiple
        copies of UDs stored in memory or on disk.  
        """

        assert name in self.landscape_rasters, "{0} is not an assigned raster".format(name)

        for indiv in self.individuals:

            # Crop the raster
            Zcrop = crop(self, indiv, name)

            Zmask = eval("Zcrop " + condition).astype(np.int64)
            
            # if hasattr(indiv, "Z"):
            #   Z = indiv.Z
            # elif hasattr(indiv, "ud_path"):
            #   Z = pd.read_pickle(indiv.ud_path)
            # else:
            #   raise AttributeError("Individual {0} does not have UD assigned".format(indiv.individual_id))

            indiv.Zmask = Zmask

            # udnew = Z * Zmask

            # if hasattr(indiv, "Z"):
            #   indiv.Z = udnew

            # # Pickle the new masked array
            # pd.to_pickle(udnew, indiv.ud_path)



    def plot_landscape(self, plot_grid=False, plot_individuals=False, 
                             by_group_size=True, plot_ud=False, 
                             plot_adjacency=False, contour=False,
                             individual_ids=True,
                             color_by_state=None):
        """
        Plot the landscape

        Parameters
        ----------
        plot_grid : bool
            Plot the grid point centers on the landscape
        plot_individuals : bool
            Plot location of individuals or groups on the landscape
        by_group_size : bool
            Plot individuals by group size or by individual location
        plot_ud : bool
            Plot UDs of the individuals
        plot_adjacency : bool
            Plot adjacency edges if they have been calculated
        contour : bool
            If plotting UDs, plot contours rather than probabilities
        individual_ids : bool
            Label individuals points with ids
        color_by_state : None or string
            Color individuals by state given by the string. Assumes individuals
            on the landscape have this property
        """

        fig, ax = plt.subplots(1, 1)
        self.polygon_gpd.plot(edgecolor='red', facecolor='white', ax=ax, zorder=-10)

        if plot_grid:
            X, Y = np.meshgrid(self.xvals, self.yvals)
            ax.plot(X, Y, 'o', ms=0.1, color='black', zorder=-1)

        if plot_individuals:

            if by_group_size:
                if hasattr(self, 'group_locations'):
                    self.group_locations.plot(markersize="group_size", ax=ax)
                else: 
                    print("Run 'assign_individuals_randomly_landscape' to plot individual locations")
            else:
                if hasattr(self, 'individuals'):
                    for indiv in self.individuals:

                        x, y = indiv.pt.coords.xy
                        ax.plot(x, y, 'o', color='red', ms=5)

                        if individual_ids:
                            ax.text(x[0], y[0], indiv.individual_id)

                        if color_by_state is not None:
                            ax.scatter(x, y, c=getattr(indiv, color_by_state), 
                                        vmin=0, vmax=1, cmap="viridis", zorder=4)

                        if plot_ud:
                            indiv.plot_ud(ax=ax, contour=contour)
        if plot_adjacency:

            if hasattr(self, "adjacency_matrix"):
                ind_ids = [indiv.individual_id for indiv in self.individuals]
                for i, host1 in enumerate(self.individuals):
                    
                    x1, y1 = host1.pt.coords.xy
                    
                    for j, host2 in enumerate(self.individuals):

                        x2, y2 = host2.pt.coords.xy
                        edge = self.adjacency_matrix[i, j]

                        if edge != 0:
                            ax.plot([x1, x2], [y1, y2], '-', color="black", zorder=-1)
            else:
                print("The 'adjacency_matrix' has not yet been created")

        return(ax)

    def assign_individuals_randomly_to_landscape(self, density, 
                                                 group_size_dist=None,
                                                 polygons=None, 
                                                 number=False,
                                                 age_structure=np.array([0.08, 0.08, 0.07, 0.07, 0.35, 0.35])):
        """
        Assign individuals to landscape with references to supplied polygons

        Parameters
        ----------
        density : float or
            Density of individuals per m2 (this must be per meter squared!)
        group_size_dist : dict
            Dictionary with 'group_size' specify the sizes of groups and 
            'prob' probability of seeing that group size. If None, assumes
            that there are no groups and individuals are just randomly placed.
        polygons : GeoSeries
            A list of Polygons. Will check whether points on the boundaries 
            also fall within the polygons before accepting a point.
        number : bool
            If True, interprets density as a number (i.e., number of deer on the landscape).
            Otherwise, it interprets it as density and scales by area appropriately
        age_structure : None or array-like
            Array of length six giving the porpotion of the population that
            are 1) weaned fawns (2 months - 1 year), female then male 
                2) yearlings (1 - 2), female then male
                3) adults (> 2), female then male

        Notes
        -----

        Assign sex based on observed sex ratios! 

        """

        if not number:
            total_individuals = np.round(self.area * density).astype(np.int64)
        else:
            total_individuals = np.int64(density)

        if group_size_dist is not None:

            # Individual centers are aligned with group
            group_sizes = group_size_dist['group_size']
            probs = group_size_dist['prob']
            self.grouped = True

        else:

            # Individuals are randomly placed
            group_sizes = np.array([1])
            probs = np.array([1])
            self.grouped = False

        # Get the group sizes based on the group size distributions
        all_groups = []
        sexes = []
        prob_male = np.sum(age_structure[np.array([1, 3, 5])])
        while np.sum(all_groups) <= total_individuals: # This slightly over assign but that is OK and makes sense given grouping
            
            all_groups.append(rand_group_size(1, group_sizes, probs)[0])
            sexes.append(np.random.choice(['M', "F"], size=1, p=[prob_male, 1 - prob_male])[0])

        all_groups = np.array(all_groups)
        sexes = np.array(sexes)

        # Assign individuals to landscape
        # NOTE: Will need to implement this differently for Shapely 2.0
        group_pts = np.array(random_points_in_polygon(self.polygon, 
                                                      len(all_groups), other_polygons=polygons), 
                                                      dtype="object")
        all_ind_locations = np.repeat(group_pts, all_groups)

        all_ind_sexes = np.repeat(sexes, all_groups)
        male_inds = all_ind_sexes == "M"
        female_inds = all_ind_sexes == "F"

        all_ind_ages = np.empty(len(all_ind_locations), dtype=np.str_)

        # Assign males and females by different stage structure
        all_ind_ages[male_inds] = np.random.choice(['F', 'Y', 'A'], 
                                        replace=True, 
                                        p=age_structure[[1, 3, 5]] / np.sum(age_structure[[1, 3, 5]]), 
                                        size=np.sum(male_inds)) 

        all_ind_ages[female_inds] = np.random.choice(['F', 'Y', 'A'], 
                                        replace=True, 
                                        p=age_structure[[0, 2, 4]] / np.sum(age_structure[[0, 2, 4]]), 
                                        size=np.sum(female_inds))    
        all_ind_groups = np.repeat(np.arange(len(all_groups)), all_groups)

        female_groups_ids = np.where(sexes == "F")[0]


        for f in range(len(all_ind_ages)):

            # What type of group
            tgroup_type = sexes[all_ind_groups[f]]
            tage = all_ind_ages[f]

            # If you are in a male group and a fawn, move to a female group
            if tgroup_type == "M" and tage == "F":
                rand_group_id = np.random.choice(female_groups_ids)
                all_ind_locations[f] = group_pts[rand_group_id]
                all_ind_groups[f] = rand_group_id

            tgroup_type = sexes[all_ind_groups[f]]

            #If you are in a female group and are a fawn, you can be male or female
            if tgroup_type == "F" and tage == "F":
                all_ind_sexes[f] = np.random.choice(["M", "F"], 
                                                    p=[1 - 0.5, 0.5])

        self.group_locations = gpd.GeoDataFrame(data={'group_size': all_groups}, geometry=group_pts)

        self.individuals = [Individual(pt, j, gid, sex=sex, age=age) for 
                            j, (pt, gid, sex, age) in enumerate(zip(all_ind_locations, all_ind_groups, all_ind_sexes, all_ind_ages))]

        self.total_individuals = len(self.individuals)
        print("A total of {0} individuals have been assigned to the landscape".format(self.total_individuals))

    def assign_known_individuals_to_landscape(self, filenames_for_uds):
        """
        Assign known individuals to the landscape. This will assign the 
        individual UDs as well.

        Parameters
        ----------
        filenames_for_uds : array-like
            Filenames specifying the information for the UDs

        """

        self.total_individuals = len(filenames_for_uds)

        individuals = []

        # Make individuals from files
        for i, flnm in enumerate(filenames_for_uds):

            animal_ud = pd.read_pickle(flnm)
            x_lower, x_upper, y_lower, y_upper = animal_ud['host_bounds']

            meanx = (x_lower + x_upper) / 2
            meany = (y_lower + y_upper) / 2

            indiv = Individual(geometry.Point((meanx, meany)), i, i)
            indiv.data_path = flnm
            individuals.append(indiv)

        self.individuals = individuals
        self.grouped = False
        self.group_locations = gpd.GeoDataFrame(data={'group_size': np.repeat(1, len(individuals))}, 
                                                geometry=[ind.pt for ind in individuals])

    def add_individual_to_landscape(self, mom_id, birth_time=np.nan, sex_ratio=0.5):
        """
        Add individual through birth for a given mom. 

        This does not assign a UD. To assign a UD, call assign_individuals_uds
        with a specific list of individuals.

        Parameters
        ----------
        mom_id : Individual object

        """

        ind_ids = [ind.individual_id for ind in self.individuals]
        max_id = np.max(ind_ids)

        # Same location and group as mom
        sex = np.random.choice(['M', "F"], size=1)[0]
        new_ind = Individual(mom_id.pt, max_id + 1, mom_id.group_id, sex=sex, age="F")
        new_ind.mom_id = mom_id.individual_id
        new_ind.birth_time = birth_time
        self.individuals.append(new_ind)

    def assign_individuals_uds(self, filenames_for_uds, ud_storage_path=None, 
                               buffer=0, Z_in_ram=False, randomize=True,
                               uds_by_sex=None, specific_individuals=None):
        """
        All individuals on the landscape are assigned UDs.  

        This is done using UDs that are already saved to disk somewhere that 
        have been empirically calculated. Each file name refers to a pickled
        object with keywords 

        'host_bounds': specifies the bounding box around the hosts movements
        'kde': The KDE estimator based on the empirical host movements

        Parameters
        ----------
        filenames_for_uds : list
            Filenames for precalculcated UDs
        ud_storage_path : string
            The directory where the temp_uds folder is made and results stored
            If None, defaults to the cwd.
        buffer : float
            Buffer the host uds
        Z_in_ram : bool
            Save the UD in RAM and disk if True. Otherwise, just save to disk.
        randomize : bool
            In True (defaul), draws random UDs.  If False, expects individuals
            to already have a data path which looks up the data needed for 
            a UD.
        uds_by_sex : None of array-like
            If array-like, it should specify whether each filename corresponds
            to a male or female
        specific_individuals : None or array of Individuals on the landscape
            Specific individuals to update UD for


        """

         # Store the UD in the cwd
        if ud_storage_path is None:
            ud_storage_path = os.getcwd()

        # Make folder to store temp UDs
        folder_path = os.path.join(ud_storage_path, "temp_uds")
        os.makedirs(folder_path, exist_ok=True)

        # Assign UDs for specific individuals or all individuals
        if specific_individuals is not None:
            individuals = specific_individuals
        else:
            individuals = self.individuals

        for indiv in individuals:
            
            # Draw an empirical UD
            outofbounds = True
            count = 0

            # This accounts for boundary issues
            while outofbounds:

                if randomize:

                    if uds_by_sex is not None:

                        # Subset on sex
                        fl_df = pd.DataFrame({'files': filenames_for_uds, 'sex': uds_by_sex})
                        tfl_df = fl_df[fl_df.sex == indiv.sex]
                        flnm = np.random.choice(tfl_df.files.values, 1)[0]
                        rand_ud = pd.read_pickle(flnm)

                    else:
                        flnm = np.random.choice(filenames_for_uds, 1)[0]
                        rand_ud = pd.read_pickle(flnm)
                else:
                    rand_ud = pd.read_pickle(indiv.data_path)

                # Extract boundaries of host movement from empirical UD
                x_lower, x_upper, y_lower, y_upper = rand_ud['host_bounds']
                xvals = np.arange(x_lower - buffer, x_upper + buffer + self.grid_size, step=self.grid_size)
                yvals = np.arange(y_lower - buffer, y_upper + buffer + self.grid_size, step=self.grid_size)


                # Get the location of the maximum prob of use
                oX, oY, oZ = rand_ud['XYZ']
                pt_ind = np.where(oZ == oZ.max()) # Get the location of max prob
                x_pt = oX[pt_ind]
                y_pt = oY[pt_ind]

                x, y = list(indiv.pt.coords)[0]
                xdiff = x - x_pt
                ydiff = y - y_pt

                # print((xdiff, ydiff))

                # Realign bounds to match the full grid
                xvals_shift, yvals_shift = realign_bounds(xvals + xdiff, yvals + ydiff, self.xvals, self.yvals)

                # Check if the new bounds are "out of bounds"
                if (len(xvals_shift) == len(xvals)) and (len(yvals_shift) == len(yvals)):
                    outofbounds = False

                # If you keep getting things that are out of bounds, eventually stop trying...something
                # is probably wrong.
                if count >= 20:
                    raise AssertionError("Struggling to find a UD for individual {0} that doesn't go over the boundary".format(indiv.individual_id))

                count = count + 1

            # Calculate UD from original values
            X, Y = np.meshgrid(xvals, yvals)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(rand_ud['kde'](positions).T, X.shape) * self.grid_size * self.grid_size  # Midpoint approximation

            save_path = os.path.join(folder_path, "ud_individual_{0}.pkl".format(indiv.individual_id))
            print(save_path)
            pd.to_pickle(Z, save_path)

            # For each indiv, assign the UD
            indiv.xvals = xvals_shift
            indiv.yvals = yvals_shift
            indiv.ud_path = save_path
            if Z_in_ram:
                indiv.Z = Z


    def assign_individuals_uds_mp(self, filenames_for_uds, ud_storage_path=None, 
                                  buffer=0, cores=1, Z_in_ram=False, 
                                  randomize=False, uds_by_sex=None, 
                                  specific_individuals=None):
        """
        All individuals on the landscape are assigned UDs.  

        This is done using UDs that are already saved to disk somewhere that 
        have been empirically calculated. Each file name refers to a pickled
        object with keywords 

        'host_bounds': specifies the bounding box around the hosts movements
        'kde': The KDE estimator based on the empirical host movements

        Parameters
        ----------
        filenames_for_uds : list
            Filenames for precalculcated UDs
        ud_storage_path : string
            The directory where the temp_uds folder is made and results stored
            If None, defaults to the cwd.
        buffer : float
            Buffer the host uds
        cores : int
            Number of cores for multi-processing
        Z_in_ram : bool
            Save the UD in RAM and disk if True. Otherwise, just save to disk.
        randomize : bool
            In True (defaul), draws random UDs.  If False, expects individuals
            to already have a data path which looks up the data needed for 
            a UD.
        uds_by_sex : 
        specific_individuals : None or array of Individuals on the landscape
            Specific individuals to update UD for

        """

         # Store the UD in the cwd
        if ud_storage_path is None:
            ud_storage_path = os.getcwd()

        # Make folder to store temp UDs
        folder_path = os.path.join(ud_storage_path, "temp_uds")
        os.makedirs(folder_path, exist_ok=True)

        if specific_individuals is not None:
            individuals = specific_individuals
        else:
            individuals = self.individuals

        pool = mp.Pool(processes=cores)
        results = [pool.apply_async(_assign_uds_mp, 
                        args=(j, indiv, filenames_for_uds, folder_path, 
                              self.grid_size, self.xvals, self.yvals, buffer, 
                              Z_in_ram, randomize, uds_by_sex, len(individuals)))
                        for j, indiv in enumerate(individuals)]
        results = [p.get() for p in results]
        pool.close()

        results.sort()

        for res in results:

            indiv = individuals[res[0]]
            indiv.xvals = res[1]
            indiv.yvals = res[2]
            indiv.ud_path = res[3]

            if Z_in_ram:
                indiv.Z = res[4]

    def remove_ud_in_memory(self):
        """
        Delete all in memory UDs to free up space
        """

        [delattr(ind, "Z") for ind in self.individuals]

    def check_individuals_have_uds(self):
        """ Check if all individuals on landscape have UDs """

        return(np.all([hasattr(indiv, "ud_path") for indiv in self.individuals]))

    def build_individual_dataframe(self):
        """ Build a matrix that is helpful for summarizing """
        pts_gpd = gpd.GeoDataFrame(geometry=[i.pt for i in self.individuals], 
                           data={'id': [i.individual_id for i in self.individuals],
                                 'sex': [i.sex for i in self.individuals],
                                 'age': [i.age for i in self.individuals],
                                 'alive': [i.alive for i in self.individuals],
                                 'group_id': [i.group_id for i in self.individuals],
                                 'birth_time': [i.birth_time for i in self.individuals],
                                 'death_time': [i.death_time for i in self.individuals]})
        return(pts_gpd)

    def individual_dispersal(self, ind_id, mean_dispersal=6, k=5, 
                         habitat_polygons=None, buffer=1000, group=True):
        """
        Shift the location of an individual based on an empirical dispersal kernel. 

        Angle of dispersal is uniform from 0 to 2pi radians.

        Group with individuals of the same sex

        Parameters
        ----------
        ind_id : int
            Numeric ID of the individual to disperse
        mean_dispersal : float
            Mean dispersal distance (units km). Default is 6 km based on Long et al. 2008 for young bucks in forested habitat
        k : float
            The scale parameter for dispersal. Default is 5 to replicate Long et al. 2005.
        habitat_polygons : GeoDataFrame or None
            If GeoDataFrame, contains the polygons to check for regarding habitat type
        group : bool
            If True, males will group with nearby males defined by buffer.  Otherwise, they will be solitary
        buffer : float
            After dispersal, look for males in buffer meters of new_pt 
            
            
        Notes
        -----
        Note that the function does not let the individual disperse beyond the bounds of the landscape
        """

        point_in_bounds = False
        point_in_habitat = False
        ud_in_bounds = False
        count = 0

        # Extract individual info
        ind = self.individuals[ind_id]
        sex = self.individuals[ind_id].sex
        old_pt = ind.pt
        xvals = ind.xvals
        yvals = ind.yvals
        
        pts_gpd = self.build_individual_dataframe()

        # while((not point_in_bounds or not point_in_habitat or not ud_in_bounds)):
        while(not (point_in_bounds and point_in_habitat and ud_in_bounds)):

            # Reset everything to False each time you try a point
            point_in_bounds = False
            point_in_habitat = False
            ud_in_bounds = False

            # Propose new point
            angle = stats.uniform.rvs(0, 2*np.pi, size=1)[0]
            scale = mean_dispersal / k
            step = stats.gamma.rvs(k, scale=scale, size=1)[0]

            xold, yold = np.r_[ind.pt.coords.xy]
            xnew = np.cos(angle)*step*1000 + xold  # Convert to meters
            ynew = np.sin(angle)*step*1000 + yold # Convert to meters
            
            # Set new group_id
            group_id = pts_gpd.group_id.max() + 1
            
            new_pt = geometry.Point((xnew, ynew))
            if self.polygon.contains(new_pt):
                point_in_bounds = True
            
            # Check that point is in the right habitat
            if habitat_polygons is None:
                point_in_habitat = True
            else:
                if habitat_polygons.contains(new_pt).any():
                    point_in_habitat = True
            
            # Join the group. Otherwise, stay at point
            if group:
                
                # Do this with some probability
                around_pt = new_pt.buffer(buffer)
                
                # Are you near another "Sex"" of the right age? 
                near_indiv = (pts_gpd.geometry.within(around_pt) & 
                              (pts_gpd.id != ind.individual_id) & 
                              (pts_gpd.sex == sex) &
                              ((pts_gpd.age == "A") | (pts_gpd.age == "Y")))
                
                # Are there any males to group up with?
                if near_indiv.any():
                    group_with_ind = np.random.choice(np.array(self.individuals)[near_indiv.values], size=1)[0]
                    new_pt = group_with_ind.pt # Use one of the points
                    group_id = group_with_ind.group_id
                       
            # Shift UD based on the new point
            oZ = pd.read_pickle(ind.ud_path)
            oX, oY = np.meshgrid(ind.xvals, ind.yvals)
            pt_ind = np.where(oZ == oZ.max()) # Get the location of max prob
            x_pt = oX[pt_ind]
            y_pt = oY[pt_ind]

            x, y = list(new_pt.coords)[0]
            xdiff = x - x_pt
            ydiff = y - y_pt

            # Realign bounds to match the full grid
            xvals_shift, yvals_shift = realign_bounds(ind.xvals + xdiff, 
                                                      ind.yvals + ydiff, 
                                                      self.xvals, self.yvals)

            # Check if point is out of bounds
            if (len(xvals_shift) == len(xvals)) and (len(yvals_shift) == len(yvals)):
                ud_in_bounds = True
            
            count = count + 1
            if count >= 100:
                habitat_polygons = None
            elif count >= 500:
                raise AssertionError("Struggling to find a pt for individual {0}".format(ind.individual_id))
        
        # Shift the individual, UD, and group_id
        ind.xvals = xvals_shift
        ind.yvals = yvals_shift
        ind.xvals_old = xvals
        ind.yvals_old = yvals
        ind.pt_old = old_pt
        ind.pt = new_pt
        ind.group_id = group_id
        ind.dispersed = True


    def get_adjacency_matrix_split(self, Z_in_ram=False,
                                   update=False, update_ids=None, append_adj=False,
                                   fixed_correlation=False, ratio_val=-10, 
                                   compare_ids=None,
                                   update_surfaces=['spatial', 'social'],
                                   wrap_landscape=False,
                                   ratio_between_group=(-1, 1),
                                   ratio_within_group=(1, 2.3)):
        """
        If there are individuals with UDs, make an adjaceny matrix
        based on UD overlap and correlation.

        Parameters
        ----------
        Z_in_ram : bool
            Save the Z matrix in RAM or just to disk 
        update : bool
            If True, expects that an adjacency matrix is already calculated.
            If False, calculates the full adjacency matrix
        update_ids : None or list of ids
            List of IDs of individuals that need an updated edge.
        append_adj : bool
            In True, expand the adjaceny matrix. Use for new individuals that 
            are born and successfully recruit
        fixed_correlation : None or float 
            Assign a fixed correlation to pairs of individuals
        compare_ids : None or list of ids
            If None, and update = True, updates the ids in update_ids for all 
            individuals on the landscape. It a list of ids, updates the ids
            in update_ids only for compare_ids. For example, if you want to
            change male-male interactions you could pass in male ids to
            update_ids and compare_ids
        wrap_landscape : False
            If False, the landscape is not wrapped. If True, assumes periodic
            boundaries and wraps the landscape. In other words, individuals on
            the east can contact individuals on the west and north can contact
            south.  Note, that this wrapping does not account for the corners.
        ratio_val : float
            The log ratio value of social / spatial contributions to FOI. 
            -np.inf means that social is contributing 0 to FOI.
        ratio_within_group : tuple
            The upper and lower bound of the log10 ratio of foi due to 
            spatial overlap and foi due to social processes for hosts within
            the same social group.  A uniform log
            ratio is drawn and then a "correlation" value is computed such that
            social and spatial processes contribute a ratio of 10**logratio to
            FOI.
        ratio_between_group : tuple
            The upper and lower bound of the log10 ratio of foi due to 
            spatial overlap and foi due to social processes for hosts in
            different social groups.  A uniform log
            ratio is drawn and then a "correlation" value is computed such that
            social and spatial processes contribute a ratio of 10**logratio to
            FOI.

        """

        assert Z_in_ram, "For speed, only implemented for Z_in_ram == True"

        if self.check_individuals_have_uds():

            ind_ids = [indiv.individual_id for indiv in self.individuals]
            if not update:
                combos = list(itertools.combinations(ind_ids, 2))
            else: 
                # Which individuals have been shifted?
                if compare_ids is None:
                    combos = list(pairs(update_ids, ind_ids))
                else:
                    combos = list(pairs(update_ids, compare_ids))

            overlaps = {}
            sd_surface = {}
            cor_surface = {}

            # TODO: Separate the spatial overlap and correlation calculation to speed
            # things up
            for j, combo in enumerate(combos):
                
                if (j) % 1000 == 0:
                    logger.info("Working on {0} of {1}".format(j + 1, len(combos)))

                host1 = self.individuals[combo[0]]
                host2 = self.individuals[combo[1]]

                if 'spatial' in update_surfaces:

                    overlap, sdsd = spatial_overlap((host1.xvals, host1.yvals), (host2.xvals, host2.yvals), 
                                                    host1.Z, host2.Z, (self.xvals_bounds, self.yvals_bounds),
                                                    self.bounds,
                                                    wrap_landscape=wrap_landscape)
                    overlaps[combo] = overlap
                    sd_surface[combo] = sdsd

                if 'social' in update_surfaces:


                    if fixed_correlation:
                        mean_cor = 10**ratio_val
                    else:
                        # Based on empirical data, assign ratios.
                        # A uniform draw on the log scale ensures and right skewed
                        # distribution on the natural scale.
                        if host1.group_id != host2.group_id:
                            # Uniform on the log scale. Skewed on the natural scale
                            mean_cor = 10**stats.uniform.rvs(loc=ratio_between_group[0], scale=ratio_between_group[1] - ratio_between_group[0], size=1)
                        else:
                            mean_cor = 10**stats.uniform.rvs(loc=ratio_within_group[0], scale=ratio_within_group[1] - ratio_within_group[0], size=1)

                    cor_surface[combo] = mean_cor

            # Build the adjacency matrix from the host overlaps
            if 'spatial' in update_surfaces:

                if update:

                    # Don't expand the adjacency matrix
                    if not append_adj:
                        adj = self.adjacency_matrix_spatial
                        adj_sd = self.adjacency_matrix_sd
                        loop1 = update_ids

                        if compare_ids is None:
                            loop2 = ind_ids
                        else:
                            loop2 = compare_ids
                    else: 
                        # Expand the adjacency matrix
                        adj = np.zeros((len(ind_ids), len(ind_ids)))
                        adj_sd = np.zeros((len(ind_ids), len(ind_ids)))
                        nold = self.adjacency_matrix_spatial.shape[0]
                        adj[:nold, :nold] = self.adjacency_matrix_spatial
                        adj_sd[:nold, :nold] = self.adjacency_matrix_sd
                        loop1 = update_ids
                        if compare_ids is None:
                            loop2 = ind_ids
                        else:
                            loop2 = compare_ids
                else:
                    adj = np.zeros((len(ind_ids), len(ind_ids)))
                    adj_sd = np.zeros((len(ind_ids), len(ind_ids)))
                    loop1 = ind_ids
                    loop2 = ind_ids

                for h1 in loop1:
                    
                    for h2 in loop2:
                        
                        if h1 != h2:
                            
                            try:
                                ov = overlaps[(h1, h2)]
                                tsd = sd_surface[(h1, h2)]
                            except KeyError:
                                ov = overlaps[(h2, h1)]
                                tsd = sd_surface[(h2, h1)]
                                
                            adj[h1, h2] = ov
                            adj[h2, h1] = ov
                            adj_sd[h1, h2] = tsd
                            adj_sd[h2, h1] = tsd

                self.adjacency_matrix_spatial = adj
                self.adjacency_matrix_sd = adj_sd


            if 'social' in update_surfaces:

                if update:

                    # Don't expand the adjacency matrix
                    if not append_adj:
                        adj_cor = self.adjacency_matrix_cor
                        loop1 = update_ids

                        if compare_ids is None:
                            loop2 = ind_ids
                        else:
                            loop2 = compare_ids
                    else: 
                        # Expand the adjacency matrix
                        adj_cor = np.zeros((len(ind_ids), len(ind_ids)))
                        nold = self.adjacency_matrix_cor.shape[0]
                        adj_cor[:nold, :nold] = self.adjacency_matrix_cor
                        loop1 = update_ids

                        if compare_ids is None:
                            loop2 = ind_ids
                        else:
                            loop2 = compare_ids
                else:
                    adj_cor = np.zeros((len(ind_ids), len(ind_ids)))
                    loop1 = ind_ids
                    loop2 = ind_ids

                for h1 in loop1:
                    
                    for h2 in loop2:
                        
                        if h1 != h2:
                            
                            try:
                                tcor = cor_surface[(h1, h2)]
                            except KeyError:
                                tcor = cor_surface[(h2, h1)]
                                
                            adj_cor[h1, h2] = tcor
                            adj_cor[h2, h1] = tcor

                self.adjacency_matrix_cor = adj_cor


            # Compute rho_bar.  This scalar ensures that relative contributions of 
            # spatial and social processes to FOI is as given by 
            # the log ratio in adjacency_matrix_cor
            self.adjacency_matrix_rho_bar = (self.adjacency_matrix_cor * self.adjacency_matrix_spatial) / self.adjacency_matrix_sd
            self.adjacency_matrix = self.adjacency_matrix_spatial + self.adjacency_matrix_sd*self.adjacency_matrix_rho_bar
            self.adjacency_matrix[np.isnan(self.adjacency_matrix)] = 0

        else:
            print("Not all individuals have UDs")


    def kill_individuals(self, time_interval, time_buffer, death_rates):
        """
        Randomly kill individuals by age class and sex over a specified time
        interval

        Parameters
        ----------
        time_interval : float
            The duration of time over which death could happen
        time_buffer : float
            This quantity is added to any death times
        death_rates : dict
            A dict that looks on the per day death rates for all age, sex combinations
        """

        age = [ind.age for ind in self.individuals]
        sex = [ind.sex for ind in self.individuals]
        alive_before = [ind.alive for ind in self.individuals]

        # Map death rates
        full_death_rates = np.array([death_rates['{0}, {1}'.format(a, s)] for a, s in zip(age, sex)])
        death_times = stats.expon.rvs(scale=1 / full_death_rates, size=len(full_death_rates))
        dead_now = death_times < time_interval
        for i, ind in enumerate(self.individuals):

            if dead_now[i] and alive_before[i]:
                ind.alive = False
                ind.death_time = time_buffer + death_times[i]

@njit
def spatial_overlap(host1, host2, Z1, Z2, land_range, land_bounds, 
                    wrap_landscape=False):
    """
    Get the UD overlap and SD*SD overlap for two hosts

    Parameters
    ----------
    host1 : Individual object, host 1

    host2 : Individual object, host 2

    land : Landscape object, for wrapping

    Z_in_ram : bool
        Whether or not Z is in RAM

    wrap_landscape : bool
        If True wrap the landscape

    Return
    ------
    : tuple
        (overlap, sd*sd*number of cells)
    """
    # Extract host values
    xvals1, yvals1 = host1
    xvals2, yvals2 = host2
    xvals_bounds, yvals_bounds = land_range

    # For computational speed, don't even deal with home ranges that
    # are longer than the landscape. 
    too_long = ((len(xvals1) >= len(xvals_bounds)) or 
                (len(yvals1) >= len(yvals_bounds)) or
                (len(xvals2) >= len(xvals_bounds)) or
                (len(yvals2) >= len(yvals_bounds)))

    if wrap_landscape and not too_long:

        # Reflect any x or y coordinates that are over the boundary to the
        # other side
        # Pass in xvals_full and yvals_full
        xlower, ylower, xupper, yupper = land_bounds
        xvals1 = wrap(xvals1, xvals_bounds, xlower, xupper)
        xvals2 = wrap(xvals2, xvals_bounds, xlower, xupper)
        yvals1 = wrap(yvals1, yvals_bounds, ylower, yupper)
        yvals2 = wrap(yvals2, yvals_bounds, ylower, yupper)

        sharedx = np.intersect1d(xvals1, xvals2)
        sharedy = np.intersect1d(yvals1, yvals2)

        if (len(sharedx) == 0) or (len(sharedy) ==  0):

            # No overlap of grids
            overlap = 0
            sdsd = 0 # Set this to one to avoid a 0 / 0 error later.  The FOI will still evaluate to 0

        else:

            xinds1 = np.where(isin(xvals1, sharedx)) [0]
            yinds1 = np.where(isin(yvals1, sharedy))[0]

            xinds2 = np.where(isin(xvals2, sharedx))[0]
            yinds2 = np.where(isin(yvals2, sharedy))[0]

            # xinds1 = np.where(np.isin(xvals1, sharedx))[0]
            # yinds1 = np.where(np.isin(yvals1, sharedy))[0]

            # xinds2 = np.where(np.isin(xvals2, sharedx))[0]
            # yinds2 = np.where(np.isin(yvals2, sharedy))[0]

            # Check this
            xy1 = meshgrid(yinds1, xinds1)
            xy2 = meshgrid(yinds2, xinds2)

            # if Z_in_ram:
            #     Z1 = host1.get_ud()
            #     Z2 = host2.get_ud()
            # else:   
            #     Z1 = pd.read_pickle(host1.ud_path)
            #     Z2 = pd.read_pickle(host2.ud_path)

            # Format like this for numba
            Zin1 = np.empty(len(yinds1)*len(xinds1))
            for i, (y, x) in enumerate(zip(np.ravel(xy1[0]), np.ravel(xy1[1]))):
                Zin1[i] = Z1[y, x]

            Zin2 = np.empty(len(yinds2)*len(xinds2))
            for i, (y, x) in enumerate(zip(np.ravel(xy2[0]), np.ravel(xy2[1]))):
                Zin2[i] = Z2[y, x]

            # Zin1 = Z1[xy1]
            # Zin2 = Z2[xy2]

            # Get the overlap surface
            overlap = np.sum(Zin1 * Zin2)
            sd1 = np.sqrt(Zin1*(1 - Zin1))
            sd2 = np.sqrt(Zin2*(1 - Zin2))
            sdsd = np.sum(sd1 * sd2)
            # sdsd = np.mean(sd1 * sd2)*Zin1.shape[0]*Zin1.shape[1]

    else:

        sharedx = np.intersect1d(xvals1, xvals2)
        sharedy = np.intersect1d(yvals1, yvals2)

        if (len(sharedx) == 0) or (len(sharedy) ==  0):

            # No overlap of grids
            overlap = 0
            sdsd = 0 # Set to 1 to avoid a 0 / 0 error later

        else:

            # Fast way
            start_x1 = np.where(xvals1 == sharedx[0])[0][0]
            end_x1 = np.where(xvals1 == sharedx[-1])[0][0]

            start_y1 = np.where(yvals1 == sharedy[0])[0][0]
            end_y1 = np.where(yvals1 == sharedy[-1])[0][0]

            start_x2 = np.where(xvals2 == sharedx[0])[0][0]
            end_x2 = np.where(xvals2 == sharedx[-1])[0][0]

            start_y2 = np.where(yvals2 == sharedy[0])[0][0]
            end_y2 = np.where(yvals2 == sharedy[-1])[0][0]

            # if Z_in_ram:
            #     Z1 = host1.get_ud()
            #     Z2 = host2.get_ud()
            # else:   
            #     Z1 = pd.read_pickle(host1.ud_path)
            #     Z2 = pd.read_pickle(host2.ud_path)

            # Fast way
            Zin1 = Z1[start_y1:(end_y1 + 1), start_x1:(end_x1 + 1)]
            Zin2 = Z2[start_y2:(end_y2 + 1), start_x2:(end_x2 + 1)]

            # Get the overlap surface
            overlap = np.sum(Zin1 * Zin2)
            sd1 = np.sqrt(Zin1*(1 - Zin1))
            sd2 = np.sqrt(Zin2*(1 - Zin2))
            sdsd = np.sum(sd1 * sd2)
    
    return((overlap, sdsd))

@njit
def isin(x, vect):

    isin_vect = np.empty(len(x), dtype=np.bool_)
    for i in range(len(x)):
        isin_vect[i] = x[i] in vect

    return(isin_vect)

@njit
def meshgrid(x, y):
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for k in range(y.size):
        for j in range(x.size):
            xx[k,j] = x[j]  # change to x[k] if indexing xy
            yy[k,j] = y[k]  # change to y[j] if indexing xy
    return((xx, yy))


@njit
def wrap(vals, vals_full, lower, upper):
    """
    Wrap values on a landscape.

    NOTE
    ----
    This function works well when the size of the UD grid is not larger than
    the size of the landscape.  When the UD can wrap multiple times around the
    landscape or start overlapping itself on the wrap, you have to make some
    decisions about what values to include.  The code below makes those decisions
    to avoid giving errors for miss matching dimensions and duplicate values,
    but this function should not be used (or it should be modified) if the UDs
    tend to be larger than the landscape.
    """

    vals_wrap = np.copy(vals)
    
    # Wrap lower
    lower_ind = vals < lower
    sum_lower_ind = np.sum(lower_ind)
    vals_wrap[lower_ind] = vals_full[(len(vals_full) - sum_lower_ind):len(vals_full)]

    # Wrap upper
    upper_ind = vals > upper
    sum_upper_ind = np.sum(upper_ind)
    vals_wrap[upper_ind] = vals_full[:sum_upper_ind]

    # Check whether the overhang is longer than the landscape
    # If over hang is not larger than the landscape, then this reduces to the
    # simpler code given above.
    # lower_ind = vals < lower
    # diff = 0
    # sum_lower_ind = np.sum(lower_ind)
    # if sum_lower_ind > len(vals_full):
    #     sum_lower_ind = len(vals_full)
    #     diff = sum_lower_ind - len(vals_full)

    # lower_ind[:diff] = False
    # vals_wrap[lower_ind] = vals_full[(len(vals_full) - sum_lower_ind + diff):len(vals_full)]

    # # Wrap upper
    # upper_ind = vals > upper
    # diff = 0
    # sum_upper_ind = np.sum(upper_ind)
    # if sum_upper_ind > len(vals_full):
    #     sum_upper_ind = len(vals_full)
    #     diff = sum_upper_ind - len(vals_full)

    # upper_ind[(len(upper_ind) - diff):len(upper_ind)] = False
    # vals_wrap[upper_ind] = vals_full[:(sum_upper_ind - diff)]

    # # Drop duplicates to help with self wrapping
    # vals_wrap = pd.Series(vals_wrap).unique()
    
    return(vals_wrap)


def _assign_uds_mp(i, indiv, filenames_for_uds, folder_path, grid_size, 
                   xvals_full, yvals_full, buffer, Z_in_ram, randomize,
                   uds_by_sex, total_individuals):
    """
    Multiprocess the assign_individuals_uds function
    """
            
    # Draw an empirical UD
    outofbounds = True
    count = 0

    # This accounts for boundary issues
    while outofbounds:

        if randomize:

            if uds_by_sex is not None:

                # Subset on sex
                fl_df = pd.DataFrame({'files': filenames_for_uds, 'sex': uds_by_sex})
                tfl_df = fl_df[fl_df.sex == indiv.sex]
                flnm = np.random.choice(tfl_df.files.values, total_individuals)[i]
                rand_ud = pd.read_pickle(flnm)

            else:
                flnm = np.random.choice(filenames_for_uds, total_individuals)[i]
                rand_ud = pd.read_pickle(flnm)
        else:
            rand_ud = pd.read_pickle(indiv.data_path)

        # if randomize:
        #   flnm = np.random.choice(filenames_for_uds, 1)[0]
        #   rand_ud = pd.read_pickle(flnm)
        # else:
        #   rand_ud = pd.read_pickle(indiv.data_path)

        # Extract boundaries of host movement from empirical UD
        x_lower, x_upper, y_lower, y_upper = rand_ud['host_bounds']
        xvals = np.arange(x_lower - buffer, x_upper + buffer + grid_size, step=grid_size)
        yvals = np.arange(y_lower - buffer, y_upper + buffer + grid_size, step=grid_size)

        # Shift boundaries based on new point
        # Center UD on area of maximum probability
        oX, oY, oZ = rand_ud['XYZ']
        pt_ind = np.where(oZ == oZ.max()) # For speed just take area of max probability
        x_pt = oX[pt_ind]
        y_pt = oY[pt_ind]

        x, y = list(indiv.pt.coords)[0]
        xdiff = x - x_pt 
        ydiff = y - y_pt 

        # Realign bounds to match the full grid
        xvals_shift, yvals_shift = realign_bounds(xvals + xdiff, yvals + ydiff, xvals_full, yvals_full)

        # Check if the new bounds are "out of bounds"
        if (len(xvals_shift) == len(xvals)) and (len(yvals_shift) == len(yvals)):
            outofbounds = False

        # If you keep getting things that are out of bounds, eventually stop trying...something
        # is probably wrong.
        if count >= 20:
            raise AssertionError("Struggling to find a UD for individual {0} that doesn't go over the boundary".format(indiv.individual_id))

        count = count + 1

    # Calculate UD from original values
    X, Y = np.meshgrid(xvals, yvals)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(rand_ud['kde'](positions).T, X.shape) * grid_size * grid_size  # Midpoint approximation

    save_path = os.path.join(folder_path, "ud_individual_{0}.pkl".format(indiv.individual_id))
    pd.to_pickle(Z, save_path)

    if Z_in_ram:
        return((i, xvals_shift, yvals_shift, save_path, Z))
    else:
        return((i, xvals_shift, yvals_shift, save_path))

    
def random_points_in_polygon(polygon, number, other_polygons=None):
    """
    Slow approach for drawing random point in polyon
    
    Parameters
    ----------
    polygon : shapely polygon
        The area in which to draw random points
    number : int
        Number of random points to draw.
    other_polygons : GeoDataFrame or None
        If GeoDataFrame, contains geometry of other polygons.  Checks 
        if points are in these polygons as well.
    
    Returns
    -------
    : list of Shapely points
    """
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        
        pnt = geometry.Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        
        if polygon.contains(pnt):

            if other_polygons is None:
                points.append(pnt)
            else:
                if other_polygons.contains(pnt).any():
                    points.append(pnt)
    return(points)


def rand_group_size(num, group_sizes, p):
    """
    Draw from a random group size distribution given by p
    
    Parameters
    ----------
    num : int
        Number of groups to draw
    group_sizes : array-like
        Vector specifying the the number of individuals in each group
    probs : array-like
        Probability of observing a group with the given size
    
    Returns
    -------
    : array-like
        Randomly drawn group sizes
    """
    
    groups = np.random.choice(group_sizes, size=num, p=p)
    return(groups)

def realign_bounds(xvals, yvals, xvals_full, yvals_full):
    """
    Shift xvals and yvals to match specified grid defined by xvals_full and yvals_full.

    All arrays should have the same step size!

    Parameters
    ----------
    xvals : array
        Sequence of values that specify the x grid of a smaller area in the bigger area
    yvals : array
        Sequence of values that specify the y grid of a smaller area in the bigger area
    xvals_full : array
        Sequence of values that specify the x grid in the larger area
    yvals_full : array
        Sequence of values that specify the y grid in the larger area

    Returns
    -------
    : xvals_new, yvals_new
        Xvals and Yvals of the smaller area shifted right and up to match the grid boundaries of the larger area

    """

    # Check bounds to make sure lower area is within larger area

    ind_greater_x = np.argmin(xvals[0] > xvals_full)
    xvals_new = xvals_full[ind_greater_x:(ind_greater_x + len(xvals))]

    ind_greater_y = np.argmin(yvals[0] > yvals_full)
    yvals_new = yvals_full[ind_greater_y:(ind_greater_y + len(yvals))]

    return((xvals_new, yvals_new))

# def place_ud_on_landscape(xvals_red, yvals_red, xvals_full, yvals_full, Z):
#   """
#   Places the UD Z with a subset of coordinates on the full landscape.
    
#   Parameters
#   ----------
#   xvals_red : array
#       The subset of xvals that span the UD
#   yvals_red : array
#       The subset of yvals that span the UD
#   xvals_full : array
#       The xvals that span the full landscape
#   yvals_full : array
#       The yvals that span the full landscape
#   Z : array
#       The UD on calculated on the gridded landscape xvals_red, yvals_red
    
#   Return
#   ------
#   : Zin
#       Dimensions of the full landscape with Z embedded.  All other values are 0. 
#   """
    
#   xin_ind = pd.Series(xvals_full).isin(xvals_red).values
#   yin_ind = pd.Series(yvals_full).isin(yvals_red).values
#   Xin, Yin = np.meshgrid(xin_ind, yin_ind)
#   Zin = (Xin*Yin).astype(np.int64).astype(np.float64)
#   Zin[Zin == 1] = np.ravel(Z)
#   return(Zin)



def movement_R0_from_avg_foi(F, gamma):
    """
    Given the FOI matrix F, compute R0

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

def movement_r_from_avg_foi(F, gamma):
    """
    Given the FOI matrix F, compute r (intrinsic pathogen growth rate)

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
    J = F + U
    r = np.max(np.abs(np.linalg.eigvals(J)))
    return((J, r, F, U))

def movement_r_from_seasonal_foi(F, gamma, deltat, timing):
    """
    Given the FOI matrix F (seasonally varying), 
    the loss of infection rate (gamma), deltat, and seasonal timing get
    the intrinsic growth rate per time interval (year, day, etc.)

    Gives your the long-run per individual production of new infected
    individuals per time interval (e.g., per day, per year). The intrinsic
    growth rate of the pathogen.

    Parameters
    ----------
    F : array
        An T by n by n array with the average FOI individual interactions.
        The columns are depositing hosts and rows are acquiring hosts.
        T indicates the number of unique seasons. This should be
    gamma : float
        The loss of infection rate
    deltat : float
        A time-step for discretizing the model
    timing : array-like
        An array of length T (same length as the first dimension of F).
        Species the number of time steps (relative to deltat) that each
        seasonal FOI lasts.

    Note
    ----
    The units of r are defined by summing timing. For example, if timing is 
    the length of a year and the parameters are all per day, 
    then dividing the estimate of r by 365 days is going to give you the
    per day rate of increase of infection.

    """

    T = F.shape[0]
    n = F.shape[1]
    Fd = 1 - np.exp(-F*deltat)
    # Fd = F*deltat
    Ud = np.diag(np.repeat(np.exp(-gamma*deltat), n))
    # Ud = np.diag(np.repeat(1 - gamma*deltat, n))

    # Loop through the seasons
    Aseason_log = np.zeros(Ud.shape)
    for t in range(T):

        tA_log = logm(Fd[t, :, :] + Ud)
        tA = timing[t]*tA_log
        Aseason_log = Aseason_log + tA

    # This is on the log scale 
    r = np.max(np.real(np.linalg.eigvals(Aseason_log)))
    return((Aseason_log, r, Fd, Ud))


def get_beta_for_r_fxn(lower_beta, upper_beta, F, gamma, deltat, timing):
    """
    Returns a fxn that takes in desired r and returns beta 

    Parameters
    ----------
    lower_beta : float
        Lower bound for beta
    upper_beta : float
        Upper bound for beta
    F : array
        An T by n by n array with the average FOI individual interactions.
        The columns are depositing hosts and rows are acquiring hosts.
        T indicates the number of unique seasons. This should be
    gamma : float
        The loss of infection rate
    deltat : float
        A time-step for discretizing the model
    timing : array-like
        An array of length T (same length as the first dimension of F).
        Species the number of time steps (relative to deltat) that each
        seasonal FOI lasts.


    Note
    ----
    Check your answer after running this as it assumes linearity!
    """

    # Calculate two rs to get the slope
    rmat_unnorm, r1, _, _ = movement_r_from_seasonal_foi(F*lower_beta, gamma, deltat, timing)
    _, r2, _, _ = movement_r_from_seasonal_foi(F*upper_beta, gamma, deltat, timing)

    # From simulations, this relationship looks linear
    slope = (r2 - r1) / (upper_beta - lower_beta)
    intercept = r1 - slope*lower_beta

    desired_beta = lambda x: (x - intercept) / slope
    return((desired_beta, rmat_unnorm))

def get_beta_for_r(lower_beta, upper_beta, desired_r, F, gamma, deltat, timing):
    """
    Returns the value of beta that gives you the desired r (intrinsic growth rate)
    value.  

    Parameters
    ----------
    lower_beta : float
        Lower bound for beta
    upper_beta : float
        Upper bound for beta
    desired_r : float
        The r value you want for your matrix
    F : array
        An T by n by n array with the average FOI individual interactions.
        The columns are depositing hosts and rows are acquiring hosts.
        T indicates the number of unique seasons. This should be
    gamma : float
        The loss of infection rate
    deltat : float
        A time-step for discretizing the model
    timing : array-like
        An array of length T (same length as the first dimension of F).
        Species the number of time steps (relative to deltat) that each
        seasonal FOI lasts.


    Note
    ----
    Check your answer after running this as it assumes linearity!
    """

    # Calculate two rs to get the slope
    r1 = movement_r_from_seasonal_foi(F*lower_beta, gamma, deltat, timing)[1]
    r2 = movement_r_from_seasonal_foi(F*upper_beta, gamma, deltat, timing)[1]

    # From simulations, this relationship looks linear
    slope = (r2 - r1) / (upper_beta - lower_beta)
    intercept = r1 - slope*lower_beta

    desired_beta = (desired_r - intercept) / slope
    return(desired_beta)


@njit
def seir_simulation_discrete_stoch(init, n, sigma, gamma, nu, foi, timesteps, 
                                  deltat, external_infection):
    """
    Simulate the SEIR model stochastically, assuming discretization of units deltat

    For speed let's pass in everything as floats
    
    Params
    ------
    init : array (dtype int)
        Rows: states (4), Columns: Individuals (n).  
        Column-wise there should be one 1 and the rest 0s for each individual.
    n : int
        Number of individuals
    sigma : float
        1 / Duration of time in exposed class
    gamma : float
        1 / Duration of time in infected class
    nu : float
        1 / duration of time in recovered (seropositive) class (day)
    foi : array
        n x n WAIFW array
    timesteps : int
        Number of time steps
    deltat : float
        Time step (in days) of the model
    external_infection : array-like
        The external infection probability experienced by all n individuals.
        Set to all zeros if you want no external infection
    
    Returns
    -------
    : array-like
        (state, number of individuals, time_steps + 1)
    """
    
    # Hold the individual-level results: State variables (SEIR) by individuals by time steps
    all_res = np.empty((4, n, timesteps + 1), dtype=np.int64)
    all_res[:, :, 0] = init

    prob_stay_e = np.exp(-sigma*deltat)
    prob_stay_i = np.exp(-gamma*deltat)
    prob_stay_r = np.exp(-nu*deltat)
    trans_mat = np.array([[1.0, 0, 0, 1.0 - prob_stay_r], # S
                          [0, prob_stay_e, 0, 0], # E
                          [0, 1.0 - prob_stay_e, prob_stay_i, 0],# I
                          [0, 0, 1.0 - prob_stay_i, prob_stay_r]]) # R

    #ragged_foi = [(foi[:, i][foi[:, i] > 0], np.where(foi[:, i] > 0)[0]) for i in range(n)]
    
    for t in range(timesteps):
        
        states_now = all_res[:, :, t]
        I = states_now[2, :] # Get the Is
    
        # Loop through individuals 
        for i in range(n):
            
            # FOI for individual i
            ind_foi = np.sum(foi[i][0] * I[foi[i][1]]) + external_infection[i] #np.sum(foi[i, :])
            
            prob_inf = 1 - np.exp(-ind_foi*deltat)

            trans_mat[0, 0] = 1 - prob_inf
            trans_mat[1, 0] = prob_inf

            # Multinomial step to update states
            ind_now = states_now[:, i]

            ind_next = np.random.multinomial(1, trans_mat[:, np.where(ind_now == 1)[0][0]])
            all_res[:, i, t + 1] = ind_next
    
    return(all_res)


@njit
def seir_simulation_discrete_stoch_time_varying(init, n, sigma, gamma, nu, 
                                                foi, t_array,
                                                timesteps, deltat, 
                                                external_infection):
    """
    Simulate the SEIR model stochastically, assuming discretization of units deltat

    Units are days, 365 days in a year
    
    Params
    ------
    init : array (dtype int)
        Rows: states (4), Columns: Individuals (n).  
        Column-wise there should be one 1 and the rest 0s for each individual.
    n : int
        Number of individuals
    sigma : float
        1 / Duration of time in exposed class
    gamma : float
        1 / Duration of time in infected class
    nu : float
        1 / duration of time in recovered (seropositive) class (day)
    foi : list of tuples
        The FOI matrix is a T x n x n WAIFW array, where T specifies the number
        of unique FOI arrays within a year.
    t_array : array
        Cumulative time array ranging from 0 to one, specifying where
        Should be of length T to match foi array.  For example,
        [0.2, 0.5, 1] indicates three seasonal arrays, one from 0-0.2, another
        from 0.2-0.5, and another from 0.5-1.
    timesteps : int
        Number of time steps
    deltat : float
        Time step (in days) of the model
    external_infection : array-like
        The external infection probability experienced by all n individuals.
        Set to all zeros if you want no external infection
    
    Returns
    -------
    : array-like
        (state, number of individuals, time_steps + 1)
    """
    
    # Hold the individual-level results: State variables (SEIR) by individuals by time steps
    all_res = np.empty((4, n, timesteps + 1), dtype=np.int64)
    all_res[:, :, 0] = init

    prob_stay_e = np.exp(-sigma*deltat)
    prob_stay_i = np.exp(-gamma*deltat)
    prob_stay_r = np.exp(-nu*deltat)
    trans_mat = np.array([[1.0, 0, 0, 1.0 - prob_stay_r], # S
                          [0, prob_stay_e, 0, 0], # E
                          [0, 1.0 - prob_stay_e, prob_stay_i, 0],# I
                          [0, 0, 1.0 - prob_stay_i, prob_stay_r]]) # R

    # Make time vector
    start_time = 0
    end_time = deltat*timesteps
    time_vect = np.arange(0, end_time + deltat, step=deltat)

    # Make ragged FOI arrays.
    ragged_foi = [[(foi[dt, :, i][foi[dt, :, i] > 0], np.where(foi[dt, :, i] > 0)[0]) for i in range(n)] for dt in range(foi.shape[0])]
    
    for t in range(timesteps):

        time_in_year = time_vect[t] / end_time # Put on 0 to 1 units
        time_index = np.argmax(time_in_year < t_array) # What season are you in?

        states_now = all_res[:, :, t]
        I = states_now[2, :] # Get the Is
    
        # Loop through individuals 
        for i in range(n):
            
            # FOI for individual i. To slower ways to do this
            # ind_foi = np.sum(foi[time_index, i, :] * I) + external_infection[i] #np.sum(foi[i, :]) # Much slower
            # ind_foi = np.sum(ragged_foi[time_index][i][0] * I[ragged_foi[time_index][i][1]]) + external_infection[i] #np.sum(foi[i, :])

            # FOI for individual i. The fastest way to to this
            ind_foi = external_infection[i]
            for j in range(len(ragged_foi[time_index][i][1])):
                ind_foi += ragged_foi[time_index][i][0][j] * I[ragged_foi[time_index][i][1][j]]
            
            prob_inf = 1 - np.exp(-ind_foi*deltat)

            trans_mat[0, 0] = 1 - prob_inf
            trans_mat[1, 0] = prob_inf

            # Multinomial step to update states
            ind_now = states_now[:, i]

            ind_next = np.random.multinomial(1, trans_mat[:, np.where(ind_now == 1)[0][0]])
            all_res[:, i, t + 1] = ind_next
    
    return(all_res)


@njit
def seir_simulation_discrete_stoch_time_varying_model0(init, n, sigma, gamma, nu, 
                                                       foi, t_array,
                                                       timesteps, deltat, 
                                                       external_infection):
    """
    Simulate the SEIR model stochastically, assuming discretization of units deltat.

    In model0, we assume FOI is the same for all individuals, which makes this
    fast to simulate.

    Units are days, 365 days in a year
    
    Params
    ------
    init : array (dtype int)
        Rows: states (4), Columns: Individuals (n).  
        Column-wise there should be one 1 and the rest 0s for each individual.
    n : int
        Number of individuals
    sigma : float
        1 / Duration of time in exposed class
    gamma : float
        1 / Duration of time in infected class
    nu : float
        1 / duration of time in recovered (seropositive) class (day)
    foi : list of tuples
        The FOI matrix is a T x n x n WAIFW array, where T specifies the number
        of unique FOI arrays within a year.
    t_array : array
        Cumulative time array ranging from 0 to one, specifying where
        Should be of length T to match foi array.  For example,
        [0.2, 0.5, 1] indicates three seasonal arrays, one from 0-0.2, another
        from 0.2-0.5, and another from 0.5-1.
    timesteps : int
        Number of time steps
    deltat : float
        Time step (in days) of the model
    external_infection : array-like
        The external infection probability experienced by all n individuals.
        Set to all zeros if you want no external infection
    
    Returns
    -------
    : array-like
        (state, number of individuals, time_steps + 1)
    """
    
    # Hold the individual-level results: State variables (SEIR) by individuals by time steps
    all_res = np.empty((4, n, timesteps + 1), dtype=np.int64)
    all_res[:, :, 0] = init

    prob_stay_e = np.exp(-sigma*deltat)
    prob_stay_i = np.exp(-gamma*deltat)
    prob_stay_r = np.exp(-nu*deltat)
    trans_mat = np.array([[1.0, 0, 0, 1.0 - prob_stay_r], # S
                          [0, prob_stay_e, 0, 0], # E
                          [0, 1.0 - prob_stay_e, prob_stay_i, 0],# I
                          [0, 0, 1.0 - prob_stay_i, prob_stay_r]]) # R

    # Make time vector
    start_time = 0
    end_time = deltat*timesteps
    time_vect = np.arange(0, end_time + deltat, step=deltat)

    # Make ragged FOI arrays.
    foi = foi[0, 1, 0] # Extract the first off diagonal
    
    for t in range(timesteps):

        time_in_year = time_vect[t] / end_time # Put on 0 to 1 units
        time_index = np.argmax(time_in_year < t_array) # What season are you in?

        states_now = all_res[:, :, t]
        I = states_now[2, :] # Get the Is
        sumI = np.sum(I)

        ind_foi = foi*sumI
            
        prob_inf = 1 - np.exp(-ind_foi*deltat)

        trans_mat[0, 0] = 1 - prob_inf
        trans_mat[1, 0] = prob_inf

        # Loop through individuals 
        for i in range(n):

            # Multinomial step to update states
            ind_now = states_now[:, i]

            ind_next = np.random.multinomial(1, trans_mat[:, np.where(ind_now == 1)[0][0]])
            all_res[:, i, t + 1] = ind_next
    
    return(all_res)



@njit
def seir_simulation_discrete_stoch_time_varying_birth_death(init, n, sigma, gamma, nu, 
                                                      foi, t_array,
                                                      timesteps, deltat, 
                                                      external_infection,
                                                      death_times, birth_times):
    """
    Simulate the SEIR model stochastically, assuming discretization of units deltat

    Units are days, 365 days in a year
    
    Params
    ------
    init : array (dtype int)
        Rows: states (4), Columns: Individuals (n).  
        Column-wise there should be one 1 and the rest 0s for each individual.
    n : int
        Number of individuals
    sigma : float
        1 / Duration of time in exposed class
    gamma : float
        1 / Duration of time in infected class
    nu : float
        1 / duration of time in recovered (seropositive) class (day)
    foi : array of arrays
        T x n x n WAIFW array, where T specifies the number of unique FOI
        arrays within a year.
    t_array : array
        Cumulative time array ranging from 0 to one, specifying where
        Should be of length T to match foi array.  For example,
        [0.2, 0.5, 1] indicates three seasonal arrays, one from 0-0.2, another
        from 0.2-0.5, and another from 0.5-1.
    timesteps : int
        Number of time steps
    deltat : float
        Time step (in days) of the model
    external_infection : array-like
        The external infection probability experienced by all n individuals.
        Set to all zeros if you want no external infection
    death_times : array-like
        Array of length n that contains the death times of hosts
    
    Returns
    -------
    : array-like
        (states + 1 (included a dead state), number of individuals, time_steps + 1)
    """
    
    # Hold the individual-level results: State variables (SEIR) by individuals by time steps
    all_res = np.zeros((6, n, timesteps + 1), dtype=np.int64)
    all_res[:, :, 0] = init

    prob_stay_e = np.exp(-sigma*deltat)
    prob_stay_i = np.exp(-gamma*deltat)
    prob_stay_r = np.exp(-nu*deltat)
    trans_mat = np.array([[1.0, 0, 0, 1.0 - prob_stay_r, 0, 0], # S
                          [0, prob_stay_e, 0, 0, 0, 0], # E
                          [0, 1.0 - prob_stay_e, prob_stay_i, 0, 0, 0],# I
                          [0, 0, 1.0 - prob_stay_i, prob_stay_r, 0, 0], # R
                          [0, 0, 0, 0, 1.0, 0], # Unborn
                          [0, 0, 0, 0, 0, 1.0]]) # Dead 

    # Make the ragged FOI
    ragged_foi = [[(foi[dt, :, i][foi[dt, :, i] > 0], np.where(foi[dt, :, i] > 0)[0]) for i in range(n)] for dt in range(foi.shape[0])]
    
    # Make time vector
    start_time = 0
    end_time = deltat*timesteps
    time_vect = np.arange(0, end_time + deltat, step=deltat)
    
    for t in range(timesteps):

        time_in_year = time_vect[t] / end_time # Put on 0 to 1 units
        time_index = np.argmax(time_in_year < t_array) # What season are you in?

        states_now = all_res[:, :, t]
        I = states_now[2, :] # Get the Is
        death_occurs = np.bitwise_and(death_times > time_vect[t], death_times <= time_vect[t + 1])  # Is the individual dead?
        birth_occurs = np.bitwise_and(birth_times > time_vect[t], birth_times <= time_vect[t + 1])

        # Loop through individuals 
        for i in range(n):

            # This is way faster than slicing! 

            # FOI for individual i. The fastest way to to this
            ind_foi = external_infection[i]
            for j in range(len(ragged_foi[time_index][i][1])):
                ind_foi += ragged_foi[time_index][i][0][j] * I[ragged_foi[time_index][i][1][j]]

            # This is much slower than a for loop in numba
            #ind_foi = np.sum(foi[time_index, i, :] * I) + external_infection[i] #np.sum(foi[i, :])
            
            prob_inf = 1 - np.exp(-ind_foi*deltat)

            # This assignment is really slow
            trans_mat[0, 0] = 1 - prob_inf
            trans_mat[1, 0] = prob_inf

            # a = 1 - prob_inf
            # b = prob_inf

            # # Multinomial step to update states
            ind_now = states_now[:, i]
            state = np.where(ind_now == 1)[0][0]
            prob_vect = trans_mat[:, state]

            if death_occurs[i]:
                # Die
                prob_vect = np.array([0, 0, 0, 0, 0, 1.0])
            elif birth_occurs[i]:
                # Born as susceptible
                prob_vect = np.array([1.0, 0, 0, 0, 0, 0.0])

            ind_next = np.random.multinomial(1, prob_vect)
            all_res[:, i, t + 1] = ind_next

    return(all_res)


@njit
def seir_simulation_discrete_stoch_time_varying_birth_death_less_memory(init, n, sigma, gamma, nu, 
                                                      foi, t_array,
                                                      timesteps, deltat, 
                                                      external_infection,
                                                      death_times, birth_times):
    """
    Simulate the SEIR model stochastically, assuming discretization of units deltat. 
    This is a memory efficient way to do the same simulation 
    as seir_simulation_discrete_stoch_time_varying_birth_death.  Far faster
    when you have many individuals.

    Units are days, 365 days in a year
    
    Params
    ------
    init : array (dtype int)
        Rows: states (6), Columns: Individuals (n).  
        Column-wise there should be one 1 and the rest 0s for each individual.
    n : int
        Number of individuals
    sigma : float
        1 / Duration of time in exposed class
    gamma : float
        1 / Duration of time in infected class
    nu : float
        1 / duration of time in recovered (seropositive) class (day)
    foi : array of arrays
        T x n x n WAIFW array, where T specifies the number of unique FOI
        arrays within a year.
    t_array : array
        Cumulative time array ranging from 0 to one, specifying where
        Should be of length T to match foi array.  For example,
        [0.2, 0.5, 1] indicates three seasonal arrays, one from 0-0.2, another
        from 0.2-0.5, and another from 0.5-1.
    timesteps : int
        Number of time steps
    deltat : float
        Time step (in days) of the model
    external_infection : array-like
        The external infection probability experienced by all n individuals.
        Set to all zeros if you want no external infection
    death_times : array-like
        Array of length n that contains the death times of hosts
    
    Returns
    -------
    : array-like
        (states + 1 (included a dead state), number of individuals, time_steps + 1)
    """
    
    # Hold the individual-level results: State variables (SEIR) by individuals by time steps
    all_res = np.empty((n, timesteps + 1), dtype=np.int64)

    # Convert init

    all_res[:, 0] = np.argmax(init, axis=0) # SEIRUD -> 0, 1, 2, 3, 4, 5 -> These are how states are labeled 

    prob_stay_e = np.exp(-sigma*deltat)
    prob_stay_i = np.exp(-gamma*deltat)
    prob_stay_r = np.exp(-nu*deltat)

    # Make the ragged FOI
    # ragged_foi = [[(foi[dt, :, i][foi[dt, :, i] > 0], np.where(foi[dt, :, i] > 0)[0]) for i in range(n)] for dt in range(foi.shape[0])]
    
    # Make time vector
    start_time = 0
    end_time = deltat*timesteps
    time_vect = np.arange(0, end_time + deltat, step=deltat)
    
    for t in range(timesteps):

        time_in_year = time_vect[t] / end_time # Put on 0 to 1 units
        time_index = np.argmax(time_in_year < t_array) # What season are you in?

        states_now = all_res[:, t]
        Iindex = np.where(states_now == 2)[0]
        death_occurs = np.bitwise_and(death_times > time_vect[t], death_times <= time_vect[t + 1])  # Is the individual dead?
        birth_occurs = np.bitwise_and(birth_times > time_vect[t], birth_times <= time_vect[t + 1])

        # Any infecteds?
        if len(Iindex) == 0:
            no_infecteds = True
        else:
            no_infecteds = False


        # Loop through individuals 
        for i in range(n):


            if death_occurs[i]:
                state_next = 5
            else:
                if birth_occurs[i] and states_now[i] == 4:

                    state_next = 0

                elif not birth_occurs[i] and states_now[i] == 4:

                    state_next = 4

                elif states_now[i] == 0: 

                    # Transmission happens
                    if not no_infecteds:
                        ind_foi = np.sum(foi[time_index, i, Iindex]) + external_infection[i] #np.sum(foi[i, :])
                        
                        prob_inf = 1 - np.exp(-ind_foi*deltat)
                    else:
                        prob_inf = external_infection[i]

                    # Individual is susceptible. Becomes exposed or stays susceptible
                    state_next =  choice([0, 1], 1 - prob_inf)

                elif states_now[i] == 1:
                    # Individual is exposed
                    state_next =  choice([1, 2], prob_stay_e)

                elif states_now[i] == 2:
                    # Individual is infected
                    state_next =  choice([2, 3], prob_stay_i)

                elif states_now[i] == 3:
                    # individual is recovered
                    state_next =  choice([3, 0], prob_stay_r)
                else:
                    # You are a dead individual, you stay dead
                    state_next = 5

            all_res[i, t + 1] = state_next

    return(all_res)

@njit
def choice(a, prob):

    r = np.random.rand()
    if r < prob:
        ret = a[0]
    else:
        ret = a[1]

    return(ret)

def get_spread_metrics(all_res, model_summary):
    """
    From a simulated individual-level trajectory, get three summaries of spatails spreas

    1. Maximum distance between infected groups
    2. Percent of groups on landscape infected
    3. Given a group is infected, average percent of individuals in the group infected

    Parameters
    ----------
    all_res : array
        4 by n by time array containing the states of all individuals through time.
    model_summary : DataFrame
        Locations and unique information for each individual on the landscape

    Returns
    -------
    : tuple
        (maximum distance between infected groups, 
         percent of groups ever infected, 
         percent infected within groups)
    """

    if type(model_summary) == list:
        model_summary = model_summary[-1] # Get the last model_summary

    # Simplify
    seir_traj = all_res.sum(axis=1)
    
    # Which individuals were infected at any point
    infected_individuals = np.any(all_res[2, :, :], axis=1)

    # Get index case
    index_case = all_res[2, :, 0]
    
    model_summary = model_summary.assign(infected = infected_individuals).assign(index_case = index_case)
    
    # What was the index group?
    idcase = np.where(model_summary.index_case == 1)[0][0]
    iddetails = model_summary.iloc[idcase, :]
    idloc = iddetails.geometry

    # Get the max distance between infected groups
    group_status = model_summary.groupby("group_id").agg({'infected' : "mean", 'geometry': lambda x: x.iloc[0]})
    infected_groups = group_status.query("infected > 0")
    infected_groups = gpd.GeoDataFrame(infected_groups)
    #max_dist_spread = infected_groups.geometry.distance(infected_groups.geometry).sort_values().max()
    max_dist = np.max(np.r_[[infected_groups.geometry.distance(x).values for x in infected_groups.geometry]].ravel())

    # Get proportion groups infected and percent infection in groups
    prop_groups_inf = np.mean(group_status.infected > 0)
    percent_infected_in_groups = infected_groups.infected.mean()

    return(max_dist, prop_groups_inf, percent_infected_in_groups)


@njit
def emergent_R0_simulation(start_index, gamma, 
                           foi,
                           group_indices=None, 
                           num_sim=100):
    """
    Calculate emergent R0 for a single individual
    
    Params
    ------
    start_index : int
        Index of individual that starts as infected
    gamma : float
        1 / Duration of time in infected class
    foi : array-like
        The FOI matrix is a n x n WAIFW array, where n is the number of individuals
    deltat : float
        Time step (in days) of the model
    num_sim : int
        Number of simulations to run for the start_index individual
    
    Returns
    -------
    : (totalR0, within group R0, between group R0)
        Each quantity in the tuple is of length num_sim
    """
    
    # Who can you infect?
    possible_cases = np.where(foi[:, start_index] > 0)[0]

    # Infection probabilities are fixed
    if group_indices is not None:
        group_id = group_indices[possible_cases]
        base_group = group_indices[start_index]
        ingroup_ind = group_id == base_group

    inf_rates = foi[start_index, possible_cases]

    num_inf = np.empty(num_sim)
    within_group = np.empty(num_sim)
    between_group = np.empty(num_sim)

    for t in range(num_sim):

        # Get the time infected.  Random draw from exponential
        time_infected = (-1 / gamma)*np.log(1 - np.random.rand())

        # Get the infection probabilities for each individual
        inf_probs = 1 - np.exp(-inf_rates*time_infected)

        # Draw whether individuals get infected
        inf_state = np.random.rand(len(possible_cases)) < inf_probs

        if group_indices is not None:
            within_group[t] = np.sum(inf_state[ingroup_ind])
            between_group[t] = np.sum(inf_state[~ingroup_ind])
            
        num_inf[t] = np.sum(inf_state)

    return((num_inf, within_group, between_group))



def emergent_R0_for_all_individuals(gamma, foi_mat, group_indices, num_sims=100):
    """
    Calculate the emergent R0 for all individuals on the landscape

    Parameters
    ----------
    gamma : float
        1 / gamma is the average duration in the infection class (units are days)
    foi_mat : array
        An n x n array that described the force of infection of who infects whom
    group_indices : array
        An array specify the group id of all n individuals in foi_mat
    num_sims : int
        The number of simulations run for each individual

    Returns
    -------
    : (total R0, within group R0, between group R0)
        total R0 = within group R0 + between group R0
    """

    allR0 = []
    allwithin = []
    allbetween = []
    for i in range(foi_mat.shape[0]):
        fullR0, withinR0, betweenR0 = emergent_R0_simulation(i, gamma, foi_mat, group_indices=group_indices, num_sim=num_sims)
        allR0.append(fullR0)
        allwithin.append(withinR0)
        allbetween.append(betweenR0)

    return((np.r_[allR0].mean(), np.r_[allwithin].mean(), np.r_[allbetween].mean()))



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


def get_hdi_polygon(Z, X, Y, percent=0.75, lower_bound=1e-10, 
                        starting_value=None, color="black"):
    """
    Find the polyon that specifies the X% HPD
    
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

    Z = Z / np.sum(Z)
    
    if starting_value is None:
        starting_value = np.mean(Z)
        
    kbest = fsolve(hdi, [starting_value], args=(Z, percent))
    Z[Z < kbest] = 0 # Mask everything else
    
    # Get contour...probably a more efficient way to do this
    cs = plt.contour(X, Y, Z, levels=[lower_bound], colors=color)
    poly = geometry.MultiPolygon([geometry.Polygon(path.vertices) for path in cs.collections[0].get_paths()])
    # verts = cs.collections[i].get_paths()[0].vertices
    # poly = geometry.Polygon(verts)
    return(poly)


def interpolate_raster_to_landscape(land, src, band=1):
    """
    Interpolate the raster to the landscape
    
    Parameters
    ----------
    land : Landscale object
        The landscape object
    src : Raster src file
    band : int
        Specifies the band to extract in the Raster source file
    
    Returns
    -------
    : The raster file interpolated onto the landscape grid
        
    """
    
    # Points to extract raster values
    X, Y = np.meshgrid(land.xvals, land.yvals)
    fullX = X.ravel()
    raster_pts = np.empty((len(fullX), 2))
    raster_pts[:, 0] = fullX
    raster_pts[:, 1] = Y.ravel()
    
    # Interpolate the raster surface...really there should be an
    # easier way to do this fast...src.sample is sooooo slow
    height = src.shape[0]
    width = src.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    x, y = rasterio.transform.xy(src.transform, rows, cols)
    x = np.ravel(np.array(x))
    y = np.ravel(np.array(y))
    z = np.ravel(src.read(band))

    xy = np.hstack([x[: , np.newaxis], 
                    y[:, np.newaxis]])
    
    # Interpolate the raster surface
    interp = NearestNDInterpolator(xy, z)
    z_new = interp(raster_pts)
    Z_new = z_new.reshape(X.shape)
    return(Z_new)


def crop(land, ind, name):
    """
    Crop the landscape to the individual
    
    Parameters
    ----------
    land : Landscape object
        The landscape
    ind : Individual object
        The individual with xvals and yvals
    name : string
        Name of the raster to crop in land.landscape_rasters
    
    Returns
    -------
    : array
        Zcrop (cropped raster as numpy array)
    """

    Zfull = land.landscape_rasters[name]
    xvalsin = pd.Series(land.xvals).isin(ind.xvals).values.astype(np.bool_)
    yvalsin = pd.Series(land.yvals).isin(ind.yvals).values.astype(np.bool_)
    Xtemp, Ytemp = np.meshgrid(xvalsin, yvalsin)
    Zmask = Xtemp * Ytemp
    Zcrop = Zfull[Zmask].reshape((len(ind.yvals), len(ind.xvals)))
    return(Zcrop)


def color_to_state(state):
    """ Map color to state """

    if state == "S":
        tcol = "blue"
    elif state == "E":
        tcol = "yellow"
    elif state == "I":
        tcol = "red"
    else:
        tcol = "gray"

    return(tcol)

color_to_state_vect = np.vectorize(color_to_state)


def pairs(*lists):
    for t in itertools.combinations(lists, 2):
        for pair in itertools.product(*t):
            yield pair


def ames_group_size_distributions():
    """
    Builds the empirical group size distributions observed during gestation
    for Ames and an adjacent area. 
    """

    # Set up group size distribution from Dailee

    # Location 5
    group_sizes5 = np.arange(1, 17)
    probs5 = np.r_[np.repeat(32 / 2, 2), 
                            np.repeat(30 / 2, 2),
                            np.repeat(16 / 5, 5),
                            np.repeat(1 / 7, 7)]
    probs5 = probs5 / np.sum(probs5)

    # Location 6
    group_sizes6 = np.arange(1, 32)
    probs6 = np.r_[np.repeat(69 / 3, 3), 
                   np.repeat(28 / 2, 2),
                   np.repeat(14 / 3, 3),
                   np.repeat(11 / 7, 7),
                   np.repeat(2 / 16, 16)]
    probs6 = probs6 / np.sum(probs6)

    samp1 = np.random.choice(group_sizes5, p=probs5, size=500000, replace=True)
    samp2 = np.random.choice(group_sizes6, p=probs6, size=500000, replace=True)
    all_samps = np.concatenate([samp1, samp2])
    freq = pd.Series(all_samps).value_counts().sort_index()

    # Combine for the distribution
    group_sizes_all = freq.index.values
    probs_all = freq.values / np.sum(freq.values)


    gsize_dist = {'group_size': group_sizes_all, 'prob': probs_all}

    return(gsize_dist)



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

    

