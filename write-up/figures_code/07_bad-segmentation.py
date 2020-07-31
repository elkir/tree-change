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

# idx=319
idx=254
rn = tc.load_run(idx,load_rast=['diff'],load_nn=False)
df = rn.df


#%%



#%%Grid of trees
plt.clf
raster = tc.chm.old
a,b = 3,6

fig,axes = plt.subplots(a,b,dpi=600)
trees = df["geometry"].sample(a*b)

for i,ax in enumerate(axes.flatten()):
    plots.plot_tree(raster,trees.iloc[i],ax=ax)
    ax.axis('off')
    ax.set_title(trees.index[i],size='x-small')

fig.tight_layout()
fig.show()

