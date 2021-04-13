import numpy as np
import pandas as pd
import os
from pathlib import Path

# vizualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap

# GIS
import fiona
import fiona.crs
import shapely
import rasterio
import rasterio.plot
import rasterio.mask
import geopandas as gpd

from src.loading import Load
import src.plots as plots
from src.mycolors import rand_cmap
from src.treechange import TreeChange,SegmentationRun

#%%
try:
    dirpath = Path(os.path.dirname(__file__))
except:
    dirpath = Path(os.getcwd())/"development_python"

dir_data = Path(f"{dirpath}/../Data/lidar/danum")

#%%

tc =TreeChange(dir_data,(2013,2014))
tc.gather_all_runs()
tc.load_rasters()

#%%
rn = tc.load_run_random(load_rast=['diff'])

#%%


thold = 34
counts = rn.df["diff_img"].apply(lambda x: np.sum(x.flatten()>thold))
#%%
fig,ax = plt.subplots(ncols=2)
colmap = plots.color_mapping_diff
arr=np.ma.masked_less_equal(tc._diff_arr,thold)
arr=arr[:arr.shape[0]//2,:arr.shape[1]//2]
cmap = colmap.cmap
cmap.set_bad('yellow')
ax[0].imshow(arr,cmap=cmap, norm = colmap.norm)
ax[1].hist(counts/rn.df['area']/4,100,log=True)
fig.show()



