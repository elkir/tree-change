import numpy as np
import pandas as pd

import os

# vizualization
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
import geopandas as gpd

from mycolors import rand_cmap

from scipy.spatial import cKDTree
#%%
print(f"cwd: {os.getcwd()}")
print(f"__filename__: {os.path.dirname(__file__)}")

#%%

years = [2013,2014]
wss = [20]


#%% Directories
try:
    dirpath = os.path.dirname(__file__)
except:
    dirpath = os.getcwd()
dir_data = f"{dirpath}/../Data/lidar"
dir_chm = f"{dir_data}/raster"
dir_treetops = f"{dir_data}/vector/treetops"
dir_crowns_r = f"{dir_data}/raster/crowns"
dir_crowns_v = f"{dir_data}/vector/crowns"

ff_index_crowns = f"{dir_crowns_r}/index.txt"
print(os.path.exists(dir_data))
#%%

# year=2013
# ws=20
# with fiona.open(filename,'r',crs=fiona.crs.from_epsg(32650)) as c:
#     print(f"Driver: {c.driver}")
#     print(f"CRS: {c.crs}")
#     print(f"Elements: {len(c)}")
#     print(f"Bounds: {c.bounds}")
#     print(f"Schema: {c.schema}")
#

#%%
## Construct a dataframe with parameters for all the runs
index_crowns = pd.read_csv(ff_index_crowns, sep=" ", names=["ws", "seed", "cr", "max"], index_col=False)
iWs20 = index_crowns['ws']==20
iSeedCr = index_crowns['seed']<index_crowns['cr']
iValid = iWs20 & iSeedCr
a=np.sum(iWs20)
b=np.sum(iValid)
print(a,b,b/a)
params = index_crowns[iValid].sample(random_state=100).iloc[0]
print(params)
#%%
def load_tt(year,ws):
    ff_tt=f"{dir_treetops}/{year}/treetops_lmf_ws{ws}.shp"
    tt = gpd.read_file(ff_tt)
    return tt

#%%
def load_cr(year,ws,params):
    f_crowns = f"dalponte_{year}_{ws}_seed{params['seed']:.5f}" \
               f"_cr{params['cr']:.6f}_max{params['max']:.3f}.json"  # TODO fix number of figures
    ff_crowns = f"{dir_crowns_v}/{year}/{f_crowns}"
    # print(os.path.exists(ff_crowns))
    cr = gpd.read_file(ff_crowns)
    cr.crs = "EPSG:32650"
    return cr

#%%

tt = [load_tt(year,wss[0]) for year in years]
cr = [load_cr(year,wss[0],params) for year in years]

#%%
def ckdnearest2(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=2)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx[:,0], gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist[:,0], name='dist'),gdB.loc[idx[:,1], gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist[:,1], name='dist2')], axis=1)
    return gdf
def ckdnearest2_alt(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=2)
    nn1 = pd.concat([gdB.loc[idx[:,0], gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist[:,0], name='dist')],axis=1)
    nn2 =pd.concat([gdB.loc[idx[:,1], gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist[:,1], name='dist2')],axis=1)
    gdf = pd.concat(
        [gdA.reset_index(drop=True),nn1,nn2], axis=1,keys=["orig","nn","nn2"])
    return gdf


#%%

nn = ckdnearest2(*tt)
diff = nn['dist2']-nn['dist']
diff = diff.rename('diff')
iDiff = diff>9
iDiff = iDiff.rename('iDiff')
crowns = pd.concat([cr[0],nn['dist'],nn['dist2'],diff,iDiff],axis=1)
#%%
rcmap = rand_cmap(1000,'soft',verbose=False)

#%%

fig,ax = plt.subplots(figsize=(16,16))
im = crowns.plot(ax=ax,column='dist',cmap="magma_r")
tt[0].plot(ax=ax,marker='.',markersize=10,color='r')
tt[1].plot(ax=ax,marker='.',markersize=10,color='b')
plt.colorbar(im.collections[0],ax=ax)
plt.show()

#%%

fig,ax = plt.subplots(figsize=(16,16))
im = crowns.plot(ax=ax,column='iDiff',cmap="Paired_r")
tt[0].plot(ax=ax,marker='.',markersize=10,color='r')
tt[1].plot(ax=ax,marker='.',markersize=10,color='b')
# plt.colorbar(im.collections[0],ax=ax)
plt.show()


