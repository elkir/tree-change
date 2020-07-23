import numpy as np
import pandas as pd
from scipy import stats
import os

# vizualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap
matplotlib.rcParams['figure.dpi'] = 300

# GIS
import fiona
import fiona.crs
import shapely
import rasterio
import rasterio.plot
import rasterio.mask
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

#%% Construct a dataframe with parameters for all the runs
index_crowns = pd.read_csv(ff_index_crowns, sep=" ", names=["ws", "seed", "cr", "max"], index_col=False)
iWs20 = index_crowns['ws']==20
iSeedCr = index_crowns['seed']<index_crowns['cr']
iValid = iWs20 & iSeedCr
a=np.sum(iWs20)
b=np.sum(iValid)
print(a,b,b/a)
params = index_crowns[iValid].sample(random_state=100).iloc[0]
print(params)
#%% load treetops from file
def load_tt(year,ws):
    '''

    :param year:
    :param ws:
    :return: GeoDataFrame with treetops as entries
    '''

    ff_tt=f"{dir_treetops}/{year}/treetops_lmf_ws{ws}.shp"
    tt = gpd.read_file(ff_tt)
    return tt

#%% load crown polygons from file
def load_cr(year,ws,params):
    '''

    :param year: int, year
    :param ws: int, window size
    :param params: parameters of dalponte, in a Series
    :return: GeoDataFrame with crown (multi)polygons
    '''
    f_crowns = f"dalponte_{year}_{ws}_seed{params['seed']:.5f}" \
               f"_cr{params['cr']:.6f}_max{params['max']:.3f}.json"  # TODO fix number of figures
    ff_crowns = f"{dir_crowns_v}/{year}/{f_crowns}"
    # print(os.path.exists(ff_crowns))
    cr = gpd.read_file(ff_crowns)
    cr['area']=cr.geometry.area
    cr.crs = "EPSG:32650"
    return cr

#%% load CHM rasters

def load_chm(year):
    ff_chm = f"{dir_chm}/raster{year}.tif"

    chm_orig = rasterio.open(ff_chm)
    crs = rasterio.crs.CRS.from_string("EPSG:32650")
    # Need a virtual dataset for overwriting CRS
    chm = rasterio.vrt.WarpedVRT(chm_orig, crs)
    return chm

#%% load all the data for both years

tt = [load_tt(year,wss[0]) for year in years]
cr = [load_cr(year,wss[0],params) for year in years]
chm = [load_chm(year) for year in years]
chm_arr = [i.read(1,masked=True) for i in chm]
#%% nearest neighbour methods
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


#%% nearest neighbour distances

nn = ckdnearest2(*tt)
diff = nn['dist2']-nn['dist']
diff = diff.rename('diff')
iDiff = diff>9
iDiff = iDiff.rename('iDiff')
crowns = pd.concat([cr[0],nn['dist'],nn['dist2'],diff,iDiff],axis=1)
#%%
rcmap = rand_cmap(1000,'soft',verbose=False)

#%% Plot absolute difference between nn and nn2

fig,ax = plt.subplots(figsize=(4,4))
im = crowns.plot(ax=ax,column='dist',cmap="magma_r")
tt[0].plot(ax=ax,marker='.',markersize=4,color='r')
tt[1].plot(ax=ax,marker='.',markersize=4,color='b')
plt.colorbar(im.collections[0],ax=ax)
plt.show()

#%% Plot trees based on threshold of nn and nn2

fig,ax = plt.subplots(figsize=(4,4))
im = crowns.plot(ax=ax,column='iDiff',cmap="Paired_r")
tt[0].plot(ax=ax,marker='.',markersize=4,color='r')
tt[1].plot(ax=ax,marker='.',markersize=4,color='b')
# plt.colorbar(im.collections[0],ax=ax)
plt.show()





#%%
trees = cr[0][cr[0]['DN']==800]
# trees = cr[0].iloc[[cr[0].geometry.area.argmax()]] # biggest tree
# trees = cr[0].iloc[[300]]

fig,ax = plt.subplots(figsize=(12,12))
im = rasterio.plot.show(chm_arr[0],transform=chm[0].transform,ax=ax)
trees.geometry.boundary.plot(ax=ax,color='r',alpha=1,linewidth=1)
# plt.colorbar(im,ax=ax)
plt.show()


#%% Get tree data and plot it
# TODO write a function that extracts a tree
out_image,out_transform = rasterio.mask.mask(chm[0],trees.geometry,crop=True,filled=False)
out_meta=chm[0].meta
out_meta.update({'driver':'GTiff',
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

rasterio.plot.show(out_image,transform=out_transform)
rasterio.plot.show_hist(out_image)
#%%

# chm_diff = chm_arr[0]-chm_arr[1]
# with rasterio.open(
#         f"{dir_chm}/diff.tif",'w',driver="GTiff",**{k: chm[0].meta[k] for k in ('width','height','nodata','crs','transform','count','dtype')}) as ds:
#     ds.write(chm_diff,1)
#%% open difference raster
chm_diff_ds = rasterio.open(f"{dir_chm}/diff.tif")
chm_diff = chm_diff_ds.read(1,masked=True)

#%% plot difference raster
fig,ax = plt.subplots(figsize=(12,12))
im= ax.imshow(chm_diff,cmap='RdBu_r',vmin=-80,vmax=80)
plt.colorbar(im,ax=ax)
plt.show()

#%% plot a single tree diference raster
out_image,out_transform = rasterio.mask.mask(chm_diff_ds,trees.geometry,crop=True,filled=False)
# out_meta=chm[0].meta
# out_meta.update({'driver':'GTiff',
#                  "height": out_image.shape[1],
#                  "width": out_image.shape[2],
#                  "transform": out_transform})

rasterio.plot.show(out_image,transform=out_transform,cmap='RdBu_r',vmin=-80,vmax=80)
rasterio.plot.show_hist(out_image)
#%%  plot a single tree histogram
plt.hist(out_image.flatten(),bins=100)
plt.show()
a= out_image
# filter outliers
q = np.quantile(out_image,[0.1,0.9])
a_reduced = a[(a>q[0]) & (a<q[1])]
a_stats = stats.describe(a_reduced)