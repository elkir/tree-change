# import numpy as np
# import pandas as pd
# from scipy import stats
import os
from pathlib import Path



# from src.loading import Load
# import src.plots as plots
# from src.mycolors import rand_cmap
from src.treechange import TreeChange



# serialization
from joblib import load, dump
import pickle

# multiprocessing
from concurrent.futures import ProcessPoolExecutor

#%%
try:
    dirpath = Path(os.path.dirname(__file__))
except:
    dirpath = Path(os.getcwd())/"development_python"

dir_data = Path(f"{dirpath}/../Data/lidar")

#%%
def run(i):
    tc = TreeChange(dir_data, (2013, 2014))
    tc.gather_all_runs()
    params = tc.runs_index.sample().iloc[0]
    tc.load_data(params)  # create_diff=True if needed
    tc.match_trees()
    tc.generate_tree_rasters(["old", "new", "diff"])
    print(f"Done {i}")

with ProcessPoolExecutor(max_workers = 8) as pool:
    for i in range(8):
        pool.submit(run,i)

## runs about 13s
#%%
#
#
# filenames= tc.runs_index.apply(lambda x: tc.load.filename_from_params(2013,x),axis=1)