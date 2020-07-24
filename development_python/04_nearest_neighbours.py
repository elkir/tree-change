import numpy as np
import pandas as pd
from osgeo import gdal,ogr,osr
gdal.UseExceptions()

# vizualization
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap

# # serialization
# from joblib import load, dump
# import pickle

# multiprocessing
from concurrent.futures import ProcessPoolExecutor

# # Example
# with ProcessPoolExecutor(max_workers = 8) as pool:
#     for file in ensembles:
#         pool.submit(full_file_routine,file)





#%%
years = [2013,2014]
ws = [20]

#%%
crs= osr.SpatialReference()
crs.ImportFromEPSG(32650)

#%%
dir_data = "Data/lidar"
dir_chm = f"{dir_data}/raster"
dir_treetops = f"{dir_data}/vector/treetops"
dir_crowns = f"{dir_data}/raster/crown"

#
# filename_crowns_index <- paste(dir_data,"raster","crowns","index.txt",sep="/")
# filename_crowns_errors <- paste(dir_data,"raster","crowns","errors.txt",sep="/")
#%%

def readCHM(year):
    chmG = gdal.Open(f"{dir_chm}/raster{year}.tif", gdal.GA_ReadOnly)
    chmG.SetSpatialRef(crs)
    return chmG

chmGDAL = [readCHM(year) for year in years]

#%%
def gdalRasterToArray(rasterGDAL):
    band = rasterGDAL.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    array = band.ReadAsArray()
    array = np.ma.masked_values(array,nodata)
    return array

chm = [gdalRasterToArray(x) for x in chmGDAL]

#%%
chm_np = np.ma.array(chm)

diff = chm_np[0]-chm_np[1]

plt.imshow(diff,cmap="RdBu");plt.colorbar();plt.show()
#%%
plt.hist(diff.flatten(),bins=100,log=True)
plt.show()
#%%
segments = 64
top = cm.get_cmap('Oranges_r', segments)
bottom = cm.get_cmap('Greens', segments)

newcolors = np.vstack((top(np.linspace(0, 1, segments)),
                       bottom(np.linspace(0, 1, segments))))
newcmp = ListedColormap(newcolors, name='OrangeGreen')
#%%
plt.imshow(np.ma.masked_inside(diff,-10,10),cmap=newcmp(4))
plt.colorbar()
plt.show()







#%%

# TODO: Shapefile loading
drv = ogr.GetDriverByName("ESRI Shapefile")


year=2013
ws=20
filename=f"{dir_treetops}/{year}/treetops_lmf_ws{ws}.shp"
ds = drv.Open(filename, 0)  # 0 means read-only. 1 means writeable.
lr = ds.GetLayer()


