
import numpy as np
import pandas as pd
from scipy import stats
import os
from pathlib import Path
# vizualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap
dpi = 300
matplotlib.rcParams['figure.dpi'] = dpi
markersize = 0.5*(300/dpi)

# GIS
import fiona
import fiona.crs
import shapely
import rasterio
import rasterio.plot
import rasterio.mask
import geopandas as gpd

from functools import partial,update_wrapper

# from mycolors import rand_cmap

from scipy.spatial import cKDTree

class Load:


    def __init__(self,dir_data):
        '''
        Constructs a data-loader utility, that contains paths to various folders and
        methods that load them based on supplied parameters.

        For custom or modified directory structure directories to them need
        to be redefined after construction.

        :param dir_data: location of the data folder (string or Path)
        '''
        self.dir_data = Path(dir_data)
        self.dir_chm = dir_data / "raster"
        self.dir_treetops = dir_data / "vector" / "treetops"
        self._dir_crowns_r = dir_data / "raster" / "crowns"
        self.dir_crowns_v = dir_data / "vector" / "crowns"
        self.ff_index_crowns = self._dir_crowns_r / "index.txt"

        self.load_chm = partial(Load.load_chm,dir=self.dir_chm)
        self.load_cr = partial(Load.load_cr,dir=self.dir_crowns_v)
        self.load_tt = partial(Load.load_tt,dir=self.dir_treetops)
        update_wrapper(self.load_chm,Load.load_chm)
        update_wrapper(self.load_cr,Load.load_cr)
        update_wrapper(self.load_tt,Load.load_tt)

    @staticmethod
    def load_chm(year,dir):
        '''
        Load CHM rasters using rasterio

        :param year:
        :param dir:
        :return: rasterio object
        '''
        ff_chm = dir/f"raster{year}.tif"

        chm_orig = rasterio.open(ff_chm)
        crs = rasterio.crs.CRS.from_string("EPSG:32650")
        # Need a virtual dataset for overwriting CRS
        chm = rasterio.vrt.WarpedVRT(chm_orig, crs)
        return chm

    @staticmethod
    def load_cr(year,ws,params,dir):
        '''
         Load crown polygons from file using Geopandas

        :param year: int, year
        :param ws: int, window size
        :param params: parameters of dalponte, in a Series
        :return: GeoDataFrame with crown (multi)polygons
        '''
        f_crowns = f"dalponte_{year}_{ws}_seed{params['seed']:.5f}" \
                   f"_cr{params['cr']:.6f}_max{params['max']:.3f}.json"  # TODO fix number of figures
        ff_crowns = dir/str(year)/f_crowns
        # print(os.path.exists(ff_crowns))
        cr = gpd.read_file(ff_crowns)
        cr['area']=cr.geometry.area
        cr.crs = "EPSG:32650"
        return cr

    @staticmethod
    def load_tt(year,ws,dir):
        '''
        Load treetops from file using geopandas.
        :param year:
        :param ws:
        :return: GeoDataFrame with treetops as entries
        '''

        ff_tt=dir/f"{year}/treetops_lmf_ws{ws}.shp"
        tt = gpd.read_file(ff_tt)
        return tt