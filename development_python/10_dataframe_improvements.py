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

from src.mycolors import rand_cmap
from src.loading import Load
from src.plots import plot_tree,plot_raster_polygon,plot_tree_hist

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
    dirpath = Path(os.getcwd())/"development_python"

dir_data = Path(f"{dirpath}/../Data/lidar/danum")
load=Load(dir_data)

ff_index_crowns = load._dir_crowns_r/"index.txt"

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

#%% load all the data for both years
tt = [load.load_tt(year,wss[0]) for year in years]
cr = [load.load_cr(year,wss[0],params) for year in years]
chm = [load.load_chm(year) for year in years]
chm_arr = [i.read(1,masked=True) for i in chm]
#%% nearest neighbour methods
def ckdnearest2(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=2)
    gdf = pd.concat(
        [gdA.reset_index(drop=True),
         gdB.loc[idx[:,0],gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist[:,0], name='dist'),
         gdB.loc[idx[:,1], gdB.columns != 'geometry'].reset_index(drop=True),
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
crowns2 = crowns
crowns2['geometry'] = crowns.geometry.exterior
    # .apply(lambda x: shapely.geometry.Polygon(x.exterior))\
    # .apply(lambda x: x.simplify(1)).exterior

del diff,iDiff
#%% Random colormap definition
rcmap = rand_cmap(1000,'bright',verbose=False)

#%% Plot absolute difference between nn and nn2
dim = 16
fig,ax = plt.subplots(figsize=(dim,dim))
im = crowns2.plot(ax=ax,column='dist',cmap=rcmap)
# im = crowns.plot(ax=ax,column='dist',cmap="magma_r")
# tt[0].plot(ax=ax,marker='+',markersize=markersize,color='r')
# tt[1].plot(ax=ax,marker='+',markersize=markersize,color='b')
plt.colorbar(im.collections[0],ax=ax)
fig.suptitle("Distance to neigbours")
plt.show()

#%% Plot trees based on threshold of nn and nn2

fig,ax = plt.subplots(figsize=(4,4))
im = crowns.plot(ax=ax,column='iDiff',cmap="Paired_r")
tt[0].plot(ax=ax,marker='+',markersize=markersize,color='r')
tt[1].plot(ax=ax,marker='+',markersize=markersize,color='b')
# plt.colorbar(im.collections[0],ax=ax)
plt.suptitle("Trees with good neighbours")
plt.show()

#%% Select some tree
# trees_df = cr[0][cr[0]['DN']==700]
trees_df = cr[0].iloc[[cr[0].geometry.area.argmax()]] # biggest tree
# trees_df = cr[0].iloc[[300]]
tree=trees_df.iloc[0].geometry

fig,ax = plt.subplots(figsize=(12,12))
im = rasterio.plot.show(chm_arr[0],transform=chm[0].transform,ax=ax)
trees_df.geometry.boundary.plot(ax=ax,color='r',alpha=1,linewidth=1)
# plt.colorbar(im,ax=ax)
plt.suptitle("Selected tree(s) (biggest?)")
fig.show()



#%% Get tree data and plot it

# TODO write a function that extracts a tree
r=chm[0]
fig,ax = plt.subplots()
plot_tree(r,tree,ax=ax)
fig.suptitle("Cutout selected tree")
fig.show()
# rasterio.plot.show_hist(out_image)


#%% write diff raster to a file
ff_diff = load.dir_chm/f"diff.tif"
# chm_diff = chm_arr[0]-chm_arr[1]
# with rasterio.open(
#         f"{load.dir_chm}/diff.tif",'w',driver="GTiff",**{k: chm[0].meta[k] for k in ('width','height','nodata','crs','transform','count','dtype')}) as ds:
#     ds.write(chm_diff,1)
if not ff_diff.exists():
    print("Difference file doesn't exist on a disk. It needs to be created.")

#%% open difference raster

chm_diff_ds = rasterio.open(ff_diff)
chm_diff = chm_diff_ds.read(1,masked=True)

#%% plot difference raster
fig,ax = plt.subplots(figsize=(12,12))
im= ax.imshow(chm_diff,cmap='RdBu_r',vmin=-80,vmax=80)
plt.colorbar(im,ax=ax)
fig.suptitle("Difference between rasters")
fig.show()

#%% plot a single tree diference raster
r=chm_diff_ds
fig,ax = plt.subplots()
plot_tree(r,tree,ax=ax,
          param_boundary=dict(color='m'),
          param_box=dict(cmap='RdBu_r',vmin=-80,vmax=80),
          param_polygon=dict(cmap='RdBu_r',vmin=-80,vmax=80))
# plt.colorbar(cax=im,ax=ax) # TODO
fig.suptitle("Selected tree on the difference raster")
fig.show()

#%%  plot a single tree histogram
plot_tree_hist(r,tree)

#%% Find change in tree height #TODO

# a= out_image
# # filter outliers
# q = np.quantile(out_image,[0.1,0.9])
# a_reduced = a[(a>q[0]) & (a<q[1])]
# a_stats = stats.describe(a_reduced)

#%% add cols to dataframe

# treeID = 6
for treeID in [300]:

    tree= tt[0][tt[0].treeID==treeID]
    crown=crowns[crowns.DN==treeID].iloc[0]
    poly = crown.geometry
    poly=poly
    print(poly)
    # plt.plot(*poly.exterior.xy)
    # if poly.interiors is not None:
    #     for ring in poly.interiors:
    #         plt.plot(*ring.xy)
    # plt.show()


#%%

(
crowns.geometry
    .apply(lambda x: shapely.geometry.Polygon(x.exterior))
    .apply(lambda x: len(x.interiors)==0)
    .describe()
)

#%%
crowns = crowns.rename(columns={"DN":"treeID"}).set_index("treeID")

#%%
# def sphericity(polygon):
#     length =
#     area =
