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

from mycolors import rand_cmap

#%%
print(f"cwd: {os.getcwd()}")
print(f"__filename__: {os.path.dirname(__file__)}")


#%%

years = [2013,2014]
wss = [20]


#%% Directories (using Pathlib)
try:
    dirpath = Path(os.path.dirname(__file__))
except:
    dirpath = Path(os.getcwd())

dir_data = Path(f"{dirpath}/../Data/lidar")
dir_chm = dir_data/"raster"
dir_treetops = dir_data/"vector"/"treetops"
dir_crowns_r = dir_data/"raster"/"crowns"
dir_crowns_v = dir_data/"vector"/"crowns"

ff_index_crowns = dir_crowns_r/"index.txt"


#%%