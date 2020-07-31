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


from src.loading import Load
import src.plots as plots
from src.mycolors import rand_cmap
from src.treechange import TreeChange


#%%
try:
    dirpath = Path(os.path.dirname(__file__))
except:
    dirpath = Path(os.getcwd())/"development_python"

dir_data = Path(f"{dirpath}/../../Data/lidar/danum")
#%%
tc = TreeChange(dir_data,(2013,2014))
tc.load_rasters()
tc.gather_all_runs()
# tc.print(folders_all=True)


#%%

idx=319
rn = tc.load_run(idx,load_rast=[],load_nn=False)


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


#%%

#%% ######### Diff colormap ##########
extent_full = shapely.geometry.box(*tc.chm.old.bounds)

(sc, x, y) =(0.1,80,0)
extent = shapely.affinity.scale(extent_full, *[sc] * 2)
extent = shapely.affinity.translate(extent, x, y)



color_mapping = plots.color_mapping_diff

arr = tc._diff_arr
arr_cutout = arr[1700:2000, 1700:2000]


#### Plot
fig = plt.figure(figsize=(8, 3),dpi=400)
widths = [5, 5, 0.2]
spec = gridspec.GridSpec(ncols=3,nrows=2,figure=fig,width_ratios=widths)
axes=[]
axes.append(fig.add_subplot(spec[:,0]))
axes.append(fig.add_subplot(spec[1,1]))
axes.append(fig.add_subplot(spec[0,1]))
axes.append(fig.add_subplot(spec[:,2]))

### Plot image
plots.plot_raster_polygon(tc.diff, extent,ax=axes[0],
                          cmap=color_mapping.cmap, norm=color_mapping.norm)
# #Old numpy cropping
# axes[0].imshow(arr_cutout, cmap=color_mapping.cmap, norm=color_mapping.norm)
axes[0].set_xticks([int(extent.bounds[0])//1+1,int(extent.bounds[2])//1])
axes[0].set_yticks([int(extent.bounds[1])//1+1,int(extent.bounds[3])//1])
axes[0].get_xaxis().get_major_formatter().set_useOffset(False)
axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
# axes[0].tick_params(axis='both', which='both',
#                     bottom=False, top=False,labelbottom=False,
#                     right=False, left=False, labelleft=False)
### Plot histogram log
hist_param = dict(x=arr.flatten(),bins=1000,log=True,density=True)
n, bins, patches = axes[1].hist(**hist_param,histtype="bar")

axes[1].hist(**hist_param, histtype="step", color='grey', linewidth=0.3)

# axes[1].set_xlim(-20,20)

# Colour histogram
bin_centers = (bins[:-1] + bins[1:]) / 2
col = color_mapping.to_rgba(bin_centers)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', c)

### Plot histogram linear
hist_param=dict(x=arr.flatten(),bins=1000, log=False,density=True)
n, bins, patches = axes[2].hist(**hist_param, histtype="bar")
axes[2].hist(**hist_param, histtype="step", color='grey', linewidth=0.3)
axes[2].set_xlim(-20, 20)

# Colour histogram
bin_centers = (bins[:-1] + bins[1:]) / 2
col = color_mapping.to_rgba(bin_centers)

for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', c)

### Plot colorbar
cbar = fig.colorbar(color_mapping, cax=axes[3], extend='both')
# cbar.minorticks_on()


fig.show()
# fig.savefig("figures/05_chm-diff.png")





