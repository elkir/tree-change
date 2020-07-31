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
rn = tc.load_run(idx,load_rast=['diff'],load_nn=False)
df = rn.df

#%%
summed = df["diff_img"].apply(lambda x: x[x>50].flatten().count())
missing_tree = df.loc[summed.sort_values().index[-4]]
extent = missing_tree.geometry.envelope
extent = shapely.affinity.scale(extent,*[1.5]*2)

#%%
extent_full = shapely.geometry.box(*tc.chm.old.bounds)



#%%

missing_tree = df.loc[summed.sort_values().index[-4]]
extent = missing_tree.geometry.envelope
(sc, x, y) =(2.5,0,0)
extent = shapely.affinity.scale(extent, *[sc] * 2)
extent = shapely.affinity.translate(extent, x, y)

#%% #########  Plot 06_missing_tree  ##########
extent_full = shapely.geometry.box(*tc.chm.old.bounds)
extent_manual = shapely.geometry.box(587423.5, 548400.375, 587508.5, 548499.125)

extent=extent_manual

arr = tc._diff_arr
arr_cutout = arr[1700:2000, 1700:2000]


colmp_chm = plots.color_mapping_chm
colmp_diff = plots.color_mapping_diff


##### Plot
### Figure and gridspec
fig = plt.figure(figsize=(6.9, 2.5),dpi=500,tight_layout=True)
widths = [0.2,5,5,5,0.2]
spec = gridspec.GridSpec(ncols=5,nrows=1,figure=fig,width_ratios=widths)

ax_cbarL = fig.add_subplot(spec[:,0])
ax_chm2013 = fig.add_subplot(spec[:,1])
ax_chm2014 = fig.add_subplot(spec[:,2])
ax_diff = fig.add_subplot(spec[:,3])
ax_cbarR = fig.add_subplot(spec[:,4])

### Plot chm 2013
plots.plot_raster_polygon(tc.chm.old, extent,ax=ax_chm2013,
                          cmap=colmp_chm.cmap, norm=colmp_chm.norm)
# #Old numpy cropping
# axes[0].imshow(arr_cutout, cmap=color_mapping.cmap, norm=color_mapping.norm)
# ax_chm2013.set_xticks([int(extent.bounds[0])//1+1,int(extent.bounds[2])//1])
# ax_chm2013.set_yticks([int(extent.bounds[1])//1+1,int(extent.bounds[3])//1])
# ax_chm2013.get_xaxis().get_major_formatter().set_useOffset(False)
# ax_chm2013.get_yaxis().get_major_formatter().set_useOffset(False)
ax_chm2013.axis('off')
ax_chm2013.set_title("CHM 2013")

### Plot chm 2014
plots.plot_raster_polygon(tc.chm.new, extent,ax=ax_chm2014,
                          cmap=colmp_chm.cmap, norm=colmp_chm.norm)
# #Old numpy cropping
# axes[0].imshow(arr_cutout, cmap=color_mapping.cmap, norm=color_mapping.norm)
ax_chm2014.set_xticks([int(extent.bounds[0])//1+1,int(extent.bounds[2])//1])
ax_chm2014.set_yticks([int(extent.bounds[1])//1+1,int(extent.bounds[3])//1])
ax_chm2014.get_xaxis().get_major_formatter().set_useOffset(False)
ax_chm2014.get_yaxis().get_major_formatter().set_useOffset(False)
ax_chm2014.axis('off')
ax_chm2014.set_title("CHM 2014")

### Plot chm diff
plots.plot_raster_polygon(tc.diff, extent,ax=ax_diff,
                          cmap=colmp_diff.cmap, norm=colmp_diff.norm)
# #Old numpy cropping
# axes[0].imshow(arr_cutout, cmap=color_mapping.cmap, norm=color_mapping.norm)
# ax_diff.set_xticks([int(extent.bounds[0])//1+1,int(extent.bounds[2])//1])
# ax_diff.set_yticks([int(extent.bounds[1])//1+1,int(extent.bounds[3])//1])
# ax_diff.get_xaxis().get_major_formatter().set_useOffset(False)
# ax_diff.get_yaxis().get_major_formatter().set_useOffset(False)
ax_diff.axis('off')
ax_diff.set_title("CHM 2014-CHM 2013")


### Plot colorbar chm - Left
cbar = fig.colorbar(colmp_chm, cax=ax_cbarL, extend='both')
ax_cbarL.yaxis.set_ticks_position('left')
# cbar.minorticks_on()

### Plot colorbar diff - Right
cbar = fig.colorbar(colmp_diff, cax=ax_cbarR, extend='both')
# cbar.minorticks_on()


### Text
xmin=round(extent.bounds[0])
ymin=round(extent.bounds[1])
xmax=round(extent.bounds[2])
ymax=round(extent.bounds[3])



text = f" coordinates: UTM 50 N    " \
       f"{xmin}–{xmax}    " \
       f"{ymin}–{ymax}"
plt.figtext(0.5,0.05,text,ha='center',size='small')



# fig.show()
fig.savefig("figures/06_missing_tree.png")
plt.close(fig)
