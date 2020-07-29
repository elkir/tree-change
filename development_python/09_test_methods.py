import os
from pathlib import Path
root = Path('/home/lune/Code/ai4er/mres')
data = Path(root/"Data"/"lidar"/"danum")

from src.loading import Load
from src.plots import plot_tree,plot_raster_polygon

import matplotlib.pyplot as plt

#%%
raster = Load.load_raster(data/"rasters"/"raster2013.tif")
crowns = Load.load_cr(2013,dict(ws=20,seed=0.2644331,cr=0.7269062,max=59.41157),dir=data/'vector'/'crowns')

treeS = crowns.iloc[200]
treeGDF = crowns.sample()



#%%

fig,ax =plt.subplots()

plot_tree(raster,treeS,ax=ax)
ax.axis('off')
fig.show()

#%%Grid of trees

fig,axes = plt.subplots(10,10)

for ax in axes.flatten():
    plot_tree(raster,crowns.sample(),ax=ax)
    ax.axis('off')
fig.tight_layout()
fig.show()




