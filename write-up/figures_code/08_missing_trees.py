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
import src.utils as utils

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
rn=tc.load_run(idx,load_rast=[],load_nn=False)
rn.find_missing_trees(pixel_ratio=0.1);


#%%
extent_full = shapely.geometry.box(*tc.chm.old.bounds)



#%% #########  Plot 06_missing_tree  ##########
extent_full = shapely.geometry.box(*tc.chm.old.bounds)
xmin,ymin,xmax,ymax =extent_full.bounds

cropx = (0.01,0.17)
cropy = (0.85,0.96)
xmin,ymin,xmax,ymax =utils.crop(extent_full.bounds,(*cropx,*cropy))


colmp_chm = plots.color_mapping_chm
colmp_diff = plots.color_mapping_diff


##### Plot
### Figure and gridspec
fig = plt.figure(figsize=(7.9, 2.5),dpi=800,constrained_layout=True)
widths = [0.2,6,6,6,0.2]
spec = gridspec.GridSpec(ncols=5,nrows=1,figure=fig,width_ratios=widths)

ax_cbarL = fig.add_subplot(spec[:,0])
ax_chm2013 = fig.add_subplot(spec[:,1])
ax_chm2014 = fig.add_subplot(spec[:,2])
ax_diff = fig.add_subplot(spec[:,3])
ax_cbarR = fig.add_subplot(spec[:,4])

### Plot chm 2013
ax,r,colmap,pol_c,title =ax_chm2013, tc.chm.old,colmp_chm,'r',"CHM 2013"

plots.plot_raster(r,ax=ax,cmap=colmap.cmap, norm=colmap.norm)
rn._tt.old.plot(marker=',',markersize=0.5,color='k',ax=ax)
polygon_par = dict()
rn.df.geometry.exterior.plot(ax=ax,color='k',
                                      linewidth=0.2,
                                      **polygon_par)
rn.df[rn.df.is_missing].exterior.plot(ax=ax,color=pol_c,
                                      linewidth=1,
                                      **polygon_par)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.axis("off")
ax.set_title(title)

### Plot chm 2014
ax,r,colmap,pol_c,title =ax_chm2014, tc.chm.new,colmp_chm,'r',"CHM 2014"

plots.plot_raster(r,ax=ax,cmap=colmap.cmap, norm=colmap.norm)
rn._tt.old.plot(marker=',',markersize=0.5,color='k',ax=ax)
polygon_par = dict()
rn.df.geometry.exterior.plot(ax=ax,color='k',
                                      linewidth=0.2,
                                      **polygon_par)
rn.df[rn.df.is_missing].exterior.plot(ax=ax,color=pol_c,
                                      linewidth=1,
                                      **polygon_par)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.axis("off")
ax.set_title(title)

### Plot chm diff
ax,r,colmap,pol_c,title =ax_diff, tc.diff,colmp_diff,'cyan',"CHM 2014-CHM 2013"

plots.plot_raster(r,ax=ax,cmap=colmap.cmap, norm=colmap.norm)
rn._tt.old.plot(marker=',',markersize=0.5,color='k',ax=ax)
polygon_par = dict()
rn.df.geometry.exterior.plot(ax=ax,color='k',
                                      linewidth=0.2,
                                      **polygon_par)
rn.df[rn.df.is_missing].exterior.plot(ax=ax,color=pol_c,
                                      linewidth=1,
                                      **polygon_par)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)
ax.axis("off")
ax.set_title(title)


### Plot colorbar chm - Left
cbar = fig.colorbar(colmp_chm, cax=ax_cbarL, extend='both')
ax_cbarL.yaxis.set_ticks_position('left')
# cbar.minorticks_on()

### Plot colorbar diff - Right
cbar = fig.colorbar(colmp_diff, cax=ax_cbarR, extend='both')
# cbar.minorticks_on()


### Text
xmin= round(xmin)
xmax= round(xmax)
ymin= round(ymin)
ymax= round(ymax)
text = f" coordinates: UTM 50 N    " \
       f"{xmin}–{xmax}    " \
       f"{ymin}–{ymax}"
plt.figtext(0.5,0.05,text,ha='center',size='small')



# fig.show()
fig.savefig("figures/08_missing_trees.png")
plt.close(fig)
