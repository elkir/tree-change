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

from scipy.spatial import cKDTree


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
print(dir_data.exists())#%%
print(f"cwd: {os.getcwd()}")
print(f"__filename__: {os.path.dirname(__file__)}")

#%%https://stackoverflow.com/questions/3842616/organizing-python-classes-in-modules-and-or-packages

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

    ff_tt=dir_treetops/str(year)/f"treetops_lmf_ws{ws}.shp"
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
    ff_crowns = dir_crowns_v/str(year)/f_crowns
    # print(os.path.exists(ff_crowns))
    cr = gpd.read_file(ff_crowns)
    cr['area']=cr.geometry.area
    cr.crs = "EPSG:32650"
    return cr

#%% load CHM rasters

def load_chm(year):
    ff_chm = dir_chm/f"raster{year}.tif"

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
crowns2['geometry'] = crowns.geometry\
    .exterior
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
plt.show()

#%% Plot trees based on threshold of nn and nn2

fig,ax = plt.subplots(figsize=(4,4))
im = crowns.plot(ax=ax,column='iDiff',cmap="Paired_r")
tt[0].plot(ax=ax,marker='+',markersize=markersize,color='r')
tt[1].plot(ax=ax,marker='+',markersize=markersize,color='b')
# plt.colorbar(im.collections[0],ax=ax)
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
plt.show()

#%%


#%% Get tree data and plot it

# TODO write a function that extracts a tree
r=chm[0]
fig,ax = plt.subplots()
box_scale=1.5
box=shapely.affinity.scale(tree.envelope,xfact=box_scale,yfact=box_scale)
gpd.GeoSeries(tree.boundary).plot(ax=ax,color='r',alpha=1,linewidth=1)
plot_raster_polygon(r,tree.simplify(1),ax=ax)
plot_raster_polygon(r,box,ax=ax,alpha=0.6)
plt.show()
# rasterio.plot.show_hist(out_image)


#%% write diff raster to a file
ff_diff = dir_chm/f"diff.tif"
# chm_diff = chm_arr[0]-chm_arr[1]
# with rasterio.open(
#         f"{dir_chm}/diff.tif",'w',driver="GTiff",**{k: chm[0].meta[k] for k in ('width','height','nodata','crs','transform','count','dtype')}) as ds:
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
plt.show()

#%% plot a single tree diference raster
r=chm_diff_ds
fig,ax = plt.subplots()
box_scale=1.5
box=shapely.affinity.scale(tree.envelope,xfact=box_scale,yfact=box_scale)
gpd.GeoSeries(tree.boundary).plot(ax=ax,color='m',alpha=0.5,linewidth=1)
im = plot_raster_polygon(r,tree,ax=ax,cmap='RdBu_r',vmin=-80,vmax=80)
plot_raster_polygon(r,box,ax=ax,alpha=0.6,cmap='RdBu_r',vmin=-80,vmax=80)
# plt.colorbar(cax=im,ax=ax) # TODO
plt.show()

# rasterio.plot.show_hist(out_image)
#%%  plot a single tree histogram
plt.hist(out_image.flatten(),bins=100)
plt.show()
a= out_image
# filter outliers
q = np.quantile(out_image,[0.1,0.9])
a_reduced = a[(a>q[0]) & (a<q[1])]
a_stats = stats.describe(a_reduced)

#%% add cols to dataframe

# treeID = 6
for treeID in range(1):

    tree= tt[0][tt[0].treeID==treeID]
    crown=crowns[crowns.DN==treeID].iloc[0]
    poly = crown.geometry
    poly=poly

    plt.plot(*poly.exterior.xy)
    if poly.interiors is not None:
        for ring in poly.interiors:
            plt.plot(*ring.xy)
    plt.show()


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
def sphericity(polygon):
    length =
    area =
