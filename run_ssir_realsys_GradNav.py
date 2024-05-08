import numpy as np
import pandas as pd
import random, os, math, tqdm, pickle
import scipy.stats as st


from realsys_utils import *
from datetime import datetime
import yaml, pickle
import argparse 
argparser = argparse.ArgumentParser()
argparser.add_argument('--dir', type=str)
argparser.add_argument('--num_step', type=int, default=2000)
argparser.add_argument('--seed', type=int, default=1)
argparser.add_argument('--last_index', type=int, default=28)

parent_dir = os.path.join( 'results/peptide/ogn', argparser.parse_args().dir)
num_step = argparser.parse_args().num_step
seed = argparser.parse_args().seed
last_index = argparser.parse_args().last_index

directory = os.path.join(parent_dir, 'ssir')
if not os.path.exists(directory):
    os.makedirs(directory)
# # ====================================================================
# read yaml file
params = yaml.safe_load(open(os.path.join(parent_dir, 'param.yaml'), 'r'))
setting = {'num_step': num_step, 'seed': seed}
with open(os.path.join(directory, 'setting.yaml'), 'w') as file:
    yaml.dump(setting, file)

####################################################################
# Set the parameters
beta = params['beta']
k = params['k']
iter_frame = num_step # total number of steps
large_batch = params['large_batch_size']
small_batch = params['small_batch_size']
contour_frac = params['boundary_frac']
threshold = params['threshold'] # distance threshold for the candidate trajectory selection
feature1_path = params['feature1_path']
feature2_path = params['feature2_path']
####################################################################


# Run Simulation
feature1 = pd.read_pickle(feature1_path) # a9
feature2 = pd.read_pickle(feature2_path) #20
traj = combine_trajectories(feature1, feature2) # all trajectories (no.1~28)
# breakpoint()
for i in tqdm.tqdm(range(1,last_index+1)):
    save_path = os.path.join(directory, f'traj_{i}')
    objSim = ObsBoundReInit_Algorithm(traj, 
                                      i, # init_traj_index
                                      beta, 
                                      k,
                                      large_batch, 
                                      small_batch, 
                                      iter_frame, 
                                      contour_frac,
                                      save_path,  
                                      threshold)
    objSim.run_simulation()