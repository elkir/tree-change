
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

from mycolors import rand_cmap

from scipy.spatial import cKDTree

class load:


# # #%% Directories (using Pathlib)
#     try:
#         dirpath = Path(os.path.dirname(__file__))
#     except:
#         dirpath = Path(os.getcwd())
#
#     dir_data = Path(f"{dirpath}/../Data/lidar")
#     dir_chm = dir_data/"raster"
#     dir_treetops = dir_data/"vector"/"treetops"
#     dir_crowns_r = dir_data/"raster"/"crowns"
#     dir_crowns_v = dir_data/"vector"/"crowns"
#
#     ff_index_crowns = dir_crowns_r/"index.txt"

    @staticmethod
    def load_tt(year,ws,dir):
        '''
        load treetops from file
        :param year:
        :param ws:
        :return: GeoDataFrame with treetops as entries
        '''

        ff_tt=dir/str(year)/f"treetops_lmf_ws{ws}.shp"
        tt = gpd.read_file(ff_tt)
        return tt

    @staticmethod
    def load_cr(year,ws,params,dir):
        '''
         load crown polygons from file

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
    def load_chm(year,dir):
        '''
        load CHM rasters

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