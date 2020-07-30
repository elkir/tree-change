import numpy as np
import pandas as pd
import scipy as sp


import os
from pathlib import Path

# vizualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib.colors as col
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
tc.load_rasters()
tc.gather_all_runs()
# tc.print(folders_all=True)


#%%

idx=319
rn = tc.load_run(idx,load_rast=[],load_nn=False)


#%%
arrCHM=tc._chm_arr.old[300:300+128,400:400+128]
arrCHM.shape

#%%
extent_full = shapely.geometry.box(*tc.chm.old.bounds)


def plot_crop(sc, x, y):
    extent = shapely.affinity.scale(extent_full, *[sc] * 2)
    extent = shapely.affinity.translate(extent, x, y)
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0)
    fig,ax = plt.subplots()
    plots.plot_raster_polygon(tc.diff, extent, norm=norm,ax=ax,cmap="PRGn_r", title=str(np.array(extent.bounds).round()))
    plt.colorbar(ax.images[0],ax=ax)
    fig.show()
    return ax


#%%

im=plot_crop(0.2,40,0)


#%% CHM colormap

cmap_CHM = plt.cm.get_cmap('viridis',64)
norm = matplotlib.colors.Normalize(0,70)
colorbar_CHM = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap_CHM)


min=0
max=60
# arr= np.random.uniform(low=min,high=max,size=(64,64))
# arr.reshape((arr.size//4,4)).sort(axis=1)
# arr.reshape((64,64))
arr=tc._chm_arr.old[:1000,:1000]


## Plot
# Plot image
fig,axes = plt.subplots(1,2,figsize=(8,3))
axes[0].imshow(arr,cmap=colorbar_CHM.cmap,norm=colorbar_CHM.norm)
plt.colorbar(colorbar_CHM,ax=axes[0])

# Plot histogram.
n, bins, patches =axes[1].hist(arr.flatten(),1000)
bin_centers = (bins[:-1] + bins[1:])/2

# scale values to interval [0,1]
col = colorbar_CHM.to_rgba(bin_centers)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', c)

fig.show()


#%% mask middle with color using set_bad
cmap = plt.cm.get_cmap('bwr',257)
newcolors = cmap(np.linspace(0, 1, 256))
cmap.set_bad('magenta')
#%% mask middle with black using set_bad


cmap = plt.cm.get_cmap('bwr',257)
newcolors = cmap(np.linspace(0, 1, 256))
#%% Diff colormap
top = cm.get_cmap('Greens_r', 128)
bottom = cm.get_cmap('RdPu', 128)

newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
cmap_diff = matplotlib.colors.ListedColormap(newcolors, name='chm_diff')
lim =50
norm = matplotlib.colors.TwoSlopeNorm(0,-lim,+lim)
color_mapping_diff = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_diff)

color_mapping=color_mapping_diff

arr=tc._diff_arr[1700:2000,1700:2000]



fig,axes = plt.subplots(1,3,figsize=(10,3))
axes[0].imshow(arr, cmap=color_mapping.cmap, norm=color_mapping.norm)


# Plot histogram.
n, bins, patches =axes[1].hist(arr.flatten(),1000,log=True,histtype="bar")
# axes[1].set_facecolor('beige')
xval= axes[1].hist(arr.flatten(),1000,log=True,histtype="step",color='grey',linewidth=0.3)

# axes[1].set_xlim(-20,20)

bin_centers = (bins[:-1] + bins[1:])/2
# scale values to interval [0,1]
col = color_mapping.to_rgba(bin_centers)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', c)
    
# Plot histogram.
n, bins, patches =axes[2].hist(arr.flatten(),1000,log=False,histtype="bar")
# axes[2].set_facecolor('beige')
xval= axes[2].hist(arr.flatten(),1000,log=False,histtype="step",color='grey',linewidth=0.3)

axes[2].set_xlim(-20,20)

bin_centers = (bins[:-1] + bins[1:])/2
# scale values to interval [0,1]
col = color_mapping.to_rgba(bin_centers)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', c)


cbar = fig.colorbar(color_mapping, use_gridspec=True,extend='both')
# cbar.minorticks_on()


fig.show()

#%%

#
#
# sample_imgs = tc.df.sample()[['old_img', 'new_img', 'diff_img']].values.flatten().tolist()
#
# fig,axes = plt.subplots(1,3)
# for i,ax in enumerate(axes):
#    ax.imshow(sample_imgs[i])
# fig.show()
#
# data = tc.df.old_img.map(np.mean).sort_values()
# plt.plot(data.values);plt.show()
#
# data =tc.df.diff_img.map(np.mean).sort_values()
# plt.plot(data.values);plt.show()
# plt.bar(range(len(data)),data.values);plt.show()

