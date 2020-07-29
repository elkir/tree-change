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


from scipy.spatial import cKDTree

from src.mycolors import rand_cmap
from src.loading import Load
import src.plots as plots
from src.mycolors import rand_cmap
from src.treechange import TreeChange

#%%
try:
    dirpath = Path(os.path.dirname(__file__))
except:
    dirpath = Path(os.getcwd())/"development_python"

dir_data = Path(f"{dirpath}/../Data/lidar/danum")
#%%
tc = TreeChange(dir_data,(2013,2014))

tc.print(folders_all=True)


#%%
print("Gather runs")
tc.gather_all_runs(print_validity=True)

#%%
print("Load and match trees")

params = tc.runs_index.sample().iloc[0]
tc.load_data(params) #create_diff=True if needed
tc.match_trees()

tc.print()


#
# print(tc._tt)
# print(tc._cr)
# print(tc.chm)

#%% Random colormap definition
rcmap = rand_cmap(1000,'bright',verbose=False)

# #%% Plot absolute difference between nn and nn2
# dim = 16
# fig,ax = plt.subplots(figsize=(dim,dim))
# im = tc.df.plot(ax=ax,column='diff',cmap='magma_r')
# # tc._tt[0].plot(ax=ax,marker='+',markersize=markersize,color='r')
# # tc._tt[1].plot(ax=ax,marker='+',markersize=markersize,color='b')
# plt.colorbar(im.collections[0],ax=ax)
# fig.suptitle("Distance to neigbours")
# plt.show()

# #%% Plot trees based on threshold of nn and nn2
#
# fig,ax = plt.subplots(figsize=(4,4))
# im = tc.df.plot(ax=ax,column='i_diff',cmap="Paired_r")
# tc._tt[0].plot(ax=ax,marker='+',markersize=markersize,color='r')
# tc._tt[1].plot(ax=ax,marker='+',markersize=markersize,color='b')
# # plt.colorbar(im.collections[0],ax=ax)
# plt.suptitle("Trees with good neighbours")
# plt.show()
#



#%% Select some tree
# trees_df = cr[0][cr[0]['DN']==700]
trees_df = tc.df.iloc[[tc.df.geometry.area.argmax()]] # biggest tree
# trees_df = cr[0].iloc[[300]]
tree=trees_df.geometry

# #%% Plot its location
# fig,ax = plt.subplots(figsize=(12,12))
# im = plots.plot_raster(tc.chm.old,ax=ax)
# gpd.GeoSeries(trees_df.geometry.boundary).plot(ax=ax,color='r',linewidth=1)
# # plt.colorbar(im,ax=ax)
# plt.suptitle("Selected tree(s) (biggest?)")
# fig.show()

#%%

#%% Get tree data and plot it

# TODO write a function that extracts a tree
r=tc.chm.old
fig,ax = plt.subplots()
plots.plot_tree(r,tree,ax=ax)
fig.suptitle("Cutout selected tree")
fig.show()
# rasterio.plot.show_hist(out_image)
#%%


#%% plot difference raster
fig,ax = plt.subplots(figsize=(12,12))

plots.plot_raster(tc.diff,ax=ax,cmap='RdBu_r',vmin=-80,vmax=80)
plt.colorbar(ax.images[0],ax=ax)
fig.suptitle("Difference between rasters")
fig.show()

#%% plot a single tree diference raster
fig,ax = plt.subplots()
plots.plot_tree(tc.diff,tree,ax=ax,
          param_boundary=dict(color='m'),
          param_box=dict(cmap='RdBu_r',vmin=-80,vmax=80),
          param_polygon=dict(cmap='RdBu_r',vmin=-80,vmax=80))
# plt.colorbar(cax=im,ax=ax) # TODO
fig.suptitle("Selected tree on the difference raster")
fig.show()

#%%  plot a single tree histogram
plt.close()
plots.plot_tree_hist(tc.diff,tree,rwidth=0.9)

#%% Find change in tree height #TODO

# # a= out_image
# # # filter outliers
# # q = np.quantile(out_image,[0.1,0.9])
# # a_reduced = a[(a>q[0]) & (a<q[1])]
# # a_stats = stats.describe(a_reduced)
#
# #%% add cols to dataframe
#
# # treeID = 6
# for treeID in [300]:
#
#     tree= tt[0][tt[0].treeID==treeID]
#     crown=crowns[crowns.DN==treeID].iloc[0]
#     poly = crown.geometry
#     poly=poly
#     print(poly)
#     # plt.plot(*poly.exterior.xy)
#     # if poly.interiors is not None:
#     #     for ring in poly.interiors:
#     #         plt.plot(*ring.xy)
#     # plt.show()



