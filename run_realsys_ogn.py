import numpy as np
import pandas as pd
import random, os, math, tqdm, pickle
import scipy.stats as st


from realsys_utils import *
from datetime import datetime
import yaml, pickle

####################################################################
# Set the parameters
random_seed = 0
random.seed(0)
beta = 0.1
k = 100
iter_frame = 3000 # total number of steps
large_batch = 100
small_batch = 30
contour_frac = 0.75
init_traj_index = 1
threshold = 0.02 # distance threshold for the candidate trajectory selection
feature1_path = 'data/fspeptide/ALA9-O_ALA9-C_ALA9-N.pkl'
feature2_path = 'data/fspeptide/ARG20-CB_ARG20-CD_ARG20-NE.pkl'


# x_limits = [1.25, 2.95]
# y_limits = [1.25, 2.95]

init_traj_index = 4 # candidate for deep well escape: 4, !6!
####################################################################

# Get the current time
current_time = datetime.now()
formatted_time = current_time.strftime('%y%m%d_%H%M')
directory = f"results/peptide/ogn/run_{formatted_time}"
if not os.path.exists(directory):
    os.makedirs(directory)

# Define all other parameters and include them in a dictionary
parameters = {
    'init_traj_index': init_traj_index,
    'random_seed': random_seed,
    'total_step': iter_frame,
    'large_batch_size': large_batch,
    'small_batch_size': small_batch,
    'k': k,
    'beta': beta,
    'boundary_frac': contour_frac,
    'feature1_path': feature1_path,
    'feature2_path': feature2_path,
}

# Serialize and save as a YAML file
with open(os.path.join(directory, 'param.yaml'), 'w') as file:
    yaml.dump(parameters, file)

# Run Simulation
feature1 = pd.read_pickle(feature1_path) # a9
feature2 = pd.read_pickle(feature2_path) #20
traj = combine_trajectories(feature1, feature2)

objSim = ObsBoundReInit_Algorithm(traj, 
                                  init_traj_index, 
                                  beta, 
                                  k,
                                  large_batch, 
                                  small_batch, 
                                  iter_frame, 
                                  contour_frac,
                                  directory,  
                                  threshold)
objSim.run_simulation()