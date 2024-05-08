import pandas as pd
import os, tqdm
from realsys_utils import *
import yaml
import argparse 

argparser = argparse.ArgumentParser()
argparser.add_argument('--dir', type=str, help='directory name')
argparser.add_argument('--num_step', type=int, default=2000, help='number of total frames (steps)')

parent_dir = os.path.join('results/peptide/GradNav', argparser.parse_args().dir)
num_step = argparser.parse_args().num_step

directory = os.path.join(parent_dir, 'ssir')
if not os.path.exists(directory):
    os.makedirs(directory)
# # ====================================================================
# read yaml file
params = yaml.safe_load(open(os.path.join(parent_dir, 'param.yaml'), 'r'))
setting = {'num_step': num_step}
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
for i in tqdm.tqdm(range(1,29)):
    save_path = os.path.join(directory, f'traj_{i}')
    objSim = GradNav_Peptide(traj, 
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