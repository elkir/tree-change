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

dir_data = Path(f"{dirpath}/../Data/lidar")
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

tc.generate_tree_rasters()
tc.print()
tc.generate_tree_rasters(["old", "new", "diff"])
tc.print()
#
# print(tc._tt)
# print(tc._cr)
# print(tc.chm)
#%%
sample_imgs = tc.df.sample()[['old_img', 'new_img', 'diff_img']].values.flatten().tolist()

fig,axes = plt.subplots(1,3)
for i,ax in enumerate(axes):
   ax.imshow(sample_imgs[i])
fig.show()

data = tc.df.old_img.map(np.mean).sort_values()
plt.plot(data.values);plt.show()

data =tc.df.diff_img.map(np.mean).sort_values()
plt.plot(data.values);plt.show()
plt.bar(range(len(data)),data.values);plt.show()



# #%% Testing masking
# tree_gdf = tc.df.iloc[[tc.df.geometry.area.argmax()]] # one (biggest) tree
# tree_gs = tree_gdf.geometry
# tree_s = tree_gdf.iloc[0]
# tree_poly = tree_s.geometry
#
# r = tc.chm.old
#
# tree_options = [tree_gdf,tree_gs,tree_s,tree_poly]
# [type(x) for x in tree_options]

#%% Raster masking development

# print("plain class")
# def mask(x):
#     try:
#         rasterio.mask.mask(r,x)
#         print(f"Yes {type(x)}")
#     except Exception as e:
#         print(f"No {type(x)} {e}")
#
# [mask(x) for x in tree_options]
# print()
# print("wrapped in dict")
# def mask2(x):
#     try:
#         rasterio.mask.mask(r,[x])
#         print(f"Yes {type(x)}")
#     except Exception as e:
#         print(f"No {type(x)} {e}")
#
# [mask2(x) for x in tree_options]

# #%% Timing cropping methods
# m,t = rasterio.mask.mask(r,tree_gdf.geometry)
# # %timeit: 84.1 ms ± 647 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# m,t = rasterio.mask.mask(r,[tree_gdf.geometry.iloc[0]])
# # %timeit: 84.2 ms ± 316 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# m,t = rasterio.mask.mask(r,tree_gdf.geometry,crop=True)
# # %timeit: 1.2 ms ± 7.21 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# m,t = rasterio.mask.mask(r,[tree_gdf.geometry.iloc[0]],crop=True)
# # %timeit: 1.16 ms ± 12.3 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# ra=tc._chm_arr.old
# msk,t,w = rasterio.mask.raster_geometry_mask(r,tree_gs); ma = np.ma.masked_array(ra,mask=m)
# # %timeit 9.11 ms ± 162 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# #%%
#
# ## Full
# m,t,w = rasterio.mask.raster_geometry_mask(r,tree_gs); ma = np.ma.masked_array(ra,mask=m)
# # %timeit m,t,w = rasterio.mask.raster_geometry_mask(r,tree_gs); ma = np.ma.masked_array(ra,mask=m)
# # 9.39 ms ± 444 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# m2,_ = rasterio.mask.mask(r,tree_gdf.geometry,filled=False,indexes=1)
# # %timeit m2,_ = rasterio.mask.mask(r,tree_gdf.geometry,filled=False,indexes=1)
# # 75.6 ms ± 13.9 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
#
# ## Cropped
# m,t,w = rasterio.mask.raster_geometry_mask(r,tree_gs,crop=True); ma = np.ma.masked_array(ra[w.toslices()],mask=m)
# # %timeit m,t,w = rasterio.mask.raster_geometry_mask(r,tree_gs,crop=True); ma = np.ma.masked_array(ra[w.toslices()],mask=m)
# # 929 µs ± 11.6 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# m2,_ = rasterio.mask.mask(r,tree_gdf.geometry,filled=False,crop=True,indexes=1)
# # %timeit m2,_ = rasterio.mask.mask(r,tree_gdf.geometry,filled=False,crop=True,indexes=1)
# # 1.12 ms ± 5.13 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
# np.array_equal(m2,ma)
#
# #%%
# def mask(poly):
#     m,t = rasterio.mask.mask(r,[poly],crop=True)
#     return m.squeeze()
#
#
# trees_gdf = tc.df.sample(9)
# trees_gs = trees_gdf.geometry
# imgs = trees_gs.map(mask)
# fig, axes = plt.subplots(3,3)
# for i,ax in enumerate(axes.flatten()):
#     ax.imshow(imgs.iloc[i])
# plt.show()


