

from collections import namedtuple
from io import StringIO

from functools import partial
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
import shapely.geometry
import rasterio
import rasterio.plot
import rasterio.mask
import geopandas as gpd


from scipy.spatial import cKDTree

from src.mycolors import rand_cmap,txt_sty
from src.loading import Load
from src.plots import plot_tree,plot_raster_polygon,plot_tree_hist
import src.utils as utils

Pair = namedtuple('Pair',"old new")


class TreeChange():
    test_var =3

    def __init__(self,dir_data,years):  #TODO infer years

        self.load = Load(dir_data)
        self.runs_index = None
        self.year = Pair(*years)
        self.chm = None
        self.diff = None
        # pure raster arrays TODO do they need to be created? A namedtuple alternative?
        self._chm_arr = None
        self._diff_arr = None
        self.runs= None

        # flags
        self.flag_interiors = None

    # def __repr__(self):
    #     #TODO with new lines
    #     ...

    def print(self,folders_all=False,runs_limit=2,):

        # define string buffer and redefine print to print to it
        buf = StringIO()
        printb = partial(print,file=buf)

        printb(f"{txt_sty['BOLD']}Tree Comparison Object{txt_sty['END']}")
        printb(f"Data (dir_data):\t{self.load.dir_data}")
        if folders_all:
            for k,v in vars(self.load).items():
                if k.startswith(('dir','ff')) and k!="dir_data":
                    try:
                        v = v.relative_to(self.load.dir_data)
                    except ValueError:
                        continue
                    printb(f"\t\t{k+':':20} {v}")

        printb(f"Years:\t{self.year or 'Not specified yet'}")

        printb(f"Rasters:")
        if self.chm is None:
            printb('Not loaded')
        else:
            for k, v in self.chm._asdict().items():
                printb(f"\t{k}: {Path(v.name).relative_to(self.load.dir_data)}")
            printb(
                f"\tdiff: {Path(self.diff.name).relative_to(self.load.dir_data) if self.diff is not None else 'unavailable'}")

        printb(f"Available runs:")
        if self.runs_index is None:
            printb(f"\tRun index not loaded")
        else:
            printb(f"\tfile: {self.load.ff_index_crowns.relative_to(self.load.dir_data)}")
            printb(f"\tNumber: {len(self.runs_index)}")
            printb(f"\tWS: {self.runs_index.ws.value_counts().sort_index().to_dict().__repr__()}")
        printb("Runs loaded:")
        n_loaded=self.runs.count()
        printb(f"\tNumber: {n_loaded}")
        if n_loaded <=0:
            ...
        else:
            sample_runs = self.runs[self.runs.notnull()]
            if n_loaded > runs_limit:
                printb("\t(Samples only)")
                sample_runs = sample_runs.sample(runs_limit)
            for k, v in sample_runs.iteritems():
                printb(f"\tRun {k}")
                printb(utils.prepend_tabs(v.print(return_string=True),2))


        #TODO check whether all status is printed
        print(buf.getvalue())



    def gather_all_runs(self,print_validity=False,valid_only=True, return_value=True):
        self.runs_index = pd.read_csv(self.load.ff_index_crowns,
                                      sep=" ", names=["ws", "seed", "cr", "max"],
                                      index_col=False)

        i_seed_less_than_cr = self.runs_index['seed'] < self.runs_index['cr']

        if print_validity:
            i_ws20 = self.runs_index['ws'] == 20
            i_valid = i_ws20 & i_seed_less_than_cr
            a = np.sum(i_ws20)
            b = np.sum(i_valid)
            print(f"Runs with ws=20: {a}")
            print(f"Of those valid: {b}")
            print(f"Ratio {b/a}")
        if valid_only:
            self.runs_index = self.runs_index[i_seed_less_than_cr].reset_index()
        print("Index of runs computed.")

        self.runs = pd.Series(index=self.runs_index.index, name="runs", dtype=object)
        return self.runs_index if return_value else None

    def load_rasters(self,create_diff=False):
        self.chm = Pair(*(self.load.load_chm(y) for y in self.year))
        self._chm_arr = Pair(*(x.read(1, masked=True) for x in self.chm))
        try:
            self.diff = self.load.load_diff()
            self._diff_arr = self.diff.read(1, masked=True)
        except FileNotFoundError:
            print("Diff didn't exist on disk and couldn't be loaded. "
                  "Set create_diff True when loading to create it.")
            if create_diff:
                self.create_diff()
                self.diff = self.load.load_diff()
                self._diff_arr = self.diff.read(1, masked=True)

    def load_run_random(self,overwrite=False,load_val = True,load_nn=True,load_rast=True):
        if overwrite:
            params = self.runs_index.sample().iloc[0]
        else:
            params = self.runs_index[self.runs.isna()].sample().iloc[0]
        run = SegmentationRun(params,self)
        if load_val:
            run.load_data()
            if load_nn:
                run.match_trees()
            if load_rast:
                run.generate_tree_rasters()
        self.runs[params.name] = run
        return run



    def load_data(self,create_diff=False,keep_interior=False):
        ...

    def create_diff(self):
        ff_diff = self.load.dir_diff / f"diff.tif"
        diff_arr = self._chm_arr.old - self._chm_arr.new

        with rasterio.open(
                ff_diff, 'w', driver="GTiff",
                **{k: self.chm[0].meta[k] for k in (
                'width', 'height', 'nodata', 'crs', 'transform', 'count', 'dtype')}
        ) as ds:
            ds.write(diff_arr, 1)

    # Expand DataFrame
    def find_missing_trees(self):
        ...



    def characterize_polygons(self):
        ...
        # area
        # perimeter
        # convex_perimeter
        # convex_area
        # major_axis
        # minor_axis
        # compactness
        # eccentricity
        # circularity
        # sphericity
        # convexity
        # curl  # fibre length?
        # solidity
        # circular_variance  # different from sphericity?
        # bending_energy
        #
        # major_axis_angle
        #
        # # requires values
        # moments



    # Plots
    def plot_raster(self):
        ...

    def plot_tree(self):
        #TODO
        ...

    def plot_tree_hist(self):
        #TODO
        ...


class SegmentationRun():

    def __init__(self,params,tc):
        self.params = params
        self.tc=tc
        self._tt = None
        self._cr = None
        self.df = None

        self.flag_interiors = tc.flag_interiors
        self.flag_neighbours = False
        ...

    def print(self,return_string=False):
        buf = StringIO()
        printb = partial(print, file=buf)

        printb(f"Parameters loaded:")
        printb(utils.prepend_tabs(f"{self.params}",1))
        printb(f"Dataframe:")
        if self.df is None:
            printb(f"\tNot constructed")
        else:
            printb(f"\tPolygon interiors: {utils.bool_to_word(self.flag_interiors)}")
            printb(f"\tRows:\t {self.df.shape[0]}")
            printb(f"\tColumns:\t {self.df.shape[1]}")
            printb(f"\t\t{self.df.columns.to_list()}")
        if return_string:
            return buf.getvalue()
        print(buf.getvalue())


    def load_data(self, keep_interior=False):
        self.flag_interiors = keep_interior
        self._tt = Pair(*(self.tc.load.load_tt(y, self.params['ws']) for y in self.tc.year))
        self._cr = Pair(*(self.tc.load.load_cr(y, self.params) for y in self.tc.year))

        self.df = self._cr.old.copy()
        self.df = self.df.rename(columns={"DN": "treeID"}).set_index("treeID")
        if not (self.flag_interiors):
            self.df['geometry'] = self.df.geometry.exterior.map(lambda x: shapely.geometry.Polygon(x))

    # Expand DataFrame
    def find_missing_trees(self):
        ...

    def match_trees(self, overwrite=False, alt=False, n=2):
        def cKDnearest2(gdfA, gdfB):
            nA = np.array(list(gdfA.geometry.apply(lambda x: (x.x, x.y))))
            nB = np.array(list(gdfB.geometry.apply(lambda x: (x.x, x.y))))
            btree = cKDTree(nB)
            dist, idx = btree.query(nA, k=2)
            gdf = pd.concat(
                [gdfA.reset_index(drop=True),
                 gdfB.loc[idx[:, 0], gdfB.columns != 'geometry'].reset_index(drop=True),
                 pd.Series(dist[:, 0], name='dist'),
                 gdfB.loc[idx[:, 1], gdfB.columns != 'geometry'].reset_index(drop=True),
                 pd.Series(dist[:, 1], name='dist2')], axis=1)
            return gdf

        # def ckdnearest2_alt(gdA, gdB):
        #     nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
        #     nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
        #     btree = cKDTree(nB)
        #     dist, idx = btree.query(nA, k=2)
        #     nn1 = pd.concat([gdB.loc[idx[:, 0], gdB.columns != 'geometry'].reset_index(drop=True),
        #                      pd.Series(dist[:, 0], name='dist')], axis=1)
        #     nn2 = pd.concat([gdB.loc[idx[:, 1], gdB.columns != 'geometry'].reset_index(drop=True),
        #                      pd.Series(dist[:, 1], name='dist2')], axis=1)
        #     gdf = pd.concat(
        #         [gdA.reset_index(drop=True), nn1, nn2], axis=1, keys=["orig", "nn", "nn2"])
        #     return gdf
        #

        # TODO
        if not(self.flag_neighbours) or overwrite:
            nn = cKDnearest2(*self._tt).set_index(self.df.index)
            diff = nn['dist2'] - nn['dist']
            i_diff = diff > 9  # TODO infer boundary
            neighbours = pd.DataFrame(dict(nn=nn['dist'], nn2=nn['dist2'], diff=diff, i_diff=i_diff))
            if overwrite:
                self.df.update(neighbours, overwrite=True)
            else:
                self.df = pd.concat([self.df, neighbours], axis=1)
                self.flag_neighbours = True

    def characterize_polygons(self):
        ...
        # area
        # perimeter
        # convex_perimeter
        # convex_area
        # major_axis
        # minor_axis
        # compactness
        # eccentricity
        # circularity
        # sphericity
        # convexity
        # curl  # fibre length?
        # solidity
        # circular_variance  # different from sphericity?
        # bending_energy
        #
        # major_axis_angle
        #
        # # requires values
        # moments
    def generate_tree_rasters(self,source=['old'],geometry=None,overwrite=False, cropped=True):

        def _mask_tree(tree,raster,raster_arr,cropped):
            m, t, w = rasterio.mask.raster_geometry_mask(raster,[tree], crop=cropped)
            if cropped:
                raster_arr = raster_arr[w.toslices()]
            m = np.logical_or(m,raster_arr.mask)
            ma = np.ma.masked_array(raster_arr, mask=m)
            return ma  #note transformation t is being discarded

        def _generate_tree_rasters_single(raster,raster_arr,raster_name,geometry,overwrite,cropped):
            col_name = f"{raster_name}_img"

            if col_name in self.df.columns and not(overwrite):
                return

            fun = lambda tree: _mask_tree(tree,raster,raster_arr,cropped)
            if geometry is None:
                geometry = self.df.geometry
            tree_rasters = geometry.map(fun)
            tree_rasters.rename(col_name,inplace=True)

            if (col_name in self.df.columns) and overwrite:
                self.df.update(tree_rasters,overwrite=True) #TODO return instead of write directly
            else:  # col_name not in df
                self.df[col_name] = tree_rasters

        keys = dict(
            old={"raster":self.tc.chm.old,"raster_arr":self.tc._chm_arr.old},
            new={"raster":self.tc.chm.new,"raster_arr":self.tc._chm_arr.new},
            diff={"raster":self.tc.diff,"raster_arr":self.tc._diff_arr}
        )
        if "all" in source:
            source = ["old","new","diff"]

        for k,v in keys.items():
            if k in source:
                _generate_tree_rasters_single(v["raster"],v["raster_arr"],k,geometry,overwrite,cropped)
        return
