
import numpy as np
import pandas as pd
import os
from pathlib import Path


# GIS
import fiona
import fiona.crs
import shapely
import rasterio
import rasterio.plot
import rasterio.mask
import geopandas as gpd

from functools import partial,update_wrapper


class Load:

    def __init__(self,dir_data):
        """
        Constructs a data-loader utility, that contains paths to various folders and
        methods that load them based on supplied parameters.

        For custom or modified directory structure directories to them need
        to be redefined after construction.

        :param dir_data: location of the data folder (string or Path)
        """
        self.dir_data = Path(dir_data).resolve()
        # self.dir_data = self.dir_data.resolve()
        self.dir_chm = self.dir_data / "raster"
        self.dir_diff = self.dir_chm
        self.dir_treetops = self.dir_data / "vector" / "treetops"
        self._dir_crowns_r = self.dir_data / "raster" / "crowns"
        self.dir_crowns_v = self.dir_data / "vector" / "crowns"
        self.ff_index_crowns = self._dir_crowns_r / "index.txt"

        self.load_cr = partial(Load.load_cr,dir=self.dir_crowns_v)
        self.load_tt = partial(Load.load_tt,dir=self.dir_treetops)
        update_wrapper(self.load_cr,Load.load_cr)
        update_wrapper(self.load_tt,Load.load_tt)

    @staticmethod
    def load_raster(filepath):
        r_orig = rasterio.open(filepath)
        #TODO is ignoring CRS when importing rasters problem?
        # crs = rasterio.crs.CRS.from_string("EPSG:32650")
        # # Need a virtual dataset for overwriting CRS
        # r = rasterio.vrt.WarpedVRT(r_orig, crs)
        #
        return r_orig


    def load_chm(self,year):
        """
        Load CHM rasters using rasterio.

        Filenaming convention rasterYYYY.tif.
        :param year:
        :return: rasterio object
        """
        return Load.load_raster(self.dir_chm/f"raster{year}.tif")

    def load_diff(self):
        return Load.load_raster(self.dir_diff/"diff.tif")





    @staticmethod
    def load_cr(year,params,dir):
        """
         Load crown polygons from file using Geopandas

        :param year: int, year
        :param params: parameters of dalponte and window size, in a Series
        :return: GeoDataFrame with crown (multi)polygons
        """

        f_crowns = Load.filename_from_params(year,params)
        ff_crowns = dir/str(year)/f_crowns
        # print(os.path.exists(ff_crowns))
        cr = gpd.read_file(ff_crowns)
        cr['area']=cr.geometry.area
        cr.crs = "EPSG:32650"
        return cr

    @staticmethod
    def load_tt(year,ws,dir):
        """
        Load treetops from file using geopandas.
        :param year:
        :param ws:
        :return: GeoDataFrame with treetops as entries
        """

        ff_tt=dir/f"{year}/treetops_lmf_ws{ws:.0f}.shp"
        tt = gpd.read_file(ff_tt)
        return tt

    @staticmethod
    def filename_from_params(year,params):

        return f"dalponte_{year}_{params['ws']:.0f}_seed{params['seed']:.5f}" \
                   f"_cr{params['cr']:.6f}_max{params['max']:.3f}.json"  # TODO fix number of figures