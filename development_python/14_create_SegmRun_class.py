# import numpy as np
# import pandas as pd
# from scipy import stats
import os
from pathlib import Path



# from src.loading import Load
# import src.plots as plots
# from src.mycolors import rand_cmap
from src.treechange import TreeChange,SegmentationRun



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
# def run(i):
#     tc = TreeChange(dir_data, (2013, 2014))
#     tc.gather_all_runs()
#     params = tc.runs_index.sample().iloc[0]
#     tc.load_data(params)  # create_diff=True if needed
#     tc.match_trees()
#     tc.generate_tree_rasters(["old", "new", "diff"])
#     print(f"Done {i}")

tc = TreeChange(dir_data, (2013, 2014))
tc.gather_all_runs()
tc.load_rasters()
params = tc.runs_index.sample().iloc[0]

#%%
run = SegmentationRun(params,tc)
run.load_data()
run.match_trees()
run.generate_tree_rasters()

# tc.print()

tc.load_run_random()
# tc.print()
