from openmm.unit import kelvin, picosecond, femtosecond, nanometer, dalton
import openmm as mm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from energy_landscape import landscape
from muller_utils import (random_init_point_with_grid,
                          get_grid_intersection_points,
                          success_cluster_identification)

import os, tqdm, pickle, yaml
from datetime import datetime
import argparse 
argparser = argparse.ArgumentParser()
argparser.add_argument('--dir', type=str)
argparser.add_argument('--interval', type=float, default=1.5)
argparser.add_argument('--num_step', type=int, default=2000)
argparser.add_argument('--seed', type=int, default=1)

parent_dir = os.path.join('results/mod_muller/LD', argparser.parse_args().dir)
interval = argparser.parse_args().interval
num_step = argparser.parse_args().num_step
seed = argparser.parse_args().seed

directory = os.path.join(parent_dir, 'ssir')
if not os.path.exists(directory):
    os.makedirs(directory)
# # ====================================================================
# read yaml file
params = yaml.safe_load(open(os.path.join(parent_dir, 'param.yaml'), 'r'))
setting = {'interval': interval, 'num_step': num_step, 'seed': seed}
with open(os.path.join(directory, 'setting.yaml'), 'w') as file:
    yaml.dump(setting, file)

# ====================================================================
# Grid parameters
x_limits = (-3.5, 2.0)
y_limits = (-1.2, 3.5)
x_limits_grid = (-3.0, 1.6)
y_limits_grid = (-1.1, 3.4)
grid_interval = (interval, interval)
center_points = np.array([[-2.7, 2.8], [-0.9, 1.3], [0.6, -0.1]])
ft = 18
e = 0.2
# Observation Guided Navigation parameters
np.random.seed(seed)
total_step = num_step #2000        # total number of steps

# Brownian simulation parameters
## each particle is totally independent, propagating under the same potential
nParticles = params['nParticles']
temp_non_dimensional = params['temperature']
mass_non_dimensional = params['mass']
friction_non_dimensional = params['friction']   
timestep_non_dimensional = params['timestep']
step = params['step']
temperature = temp_non_dimensional * kelvin
mass = mass_non_dimensional * dalton
friction = friction_non_dimensional / picosecond
timestep = timestep_non_dimensional * femtosecond

# ======================================================================

# Get initial points on the grid
_, grid_x, grid_y = random_init_point_with_grid(x_limits_grid, y_limits_grid, grid_interval=grid_interval)
initial_points = get_grid_intersection_points(grid_x, grid_y)

# Brownian simulation set-up
system = mm.System()
pes = landscape('Modified_Muller')
#plt.close()
for i in range(nParticles):
    system.addParticle(mass)
    pes.addParticle(i, [])
system.addForce(pes)

integrator = mm.LangevinIntegrator(temperature, friction, timestep)
context = mm.Context(system, integrator)

# traj_all_save = []
init_no = 1
total_num_success = 0
total_num_clusters = 0
valid_init = []
#breakpoint()
for init in tqdm.tqdm(initial_points):
    init = init.reshape(1, 3)
    startingPositions = init

    context.setPositions(startingPositions)
    context.setVelocitiesToTemperature(temperature)

    traj = np.zeros((total_step, 3))
    for i in range(total_step):
        # Assuming context, pes, and integrator are predefined and properly set up
        x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)   
        p = x[0]
        E = pes.potential(x=p[0], y=p[1])  # Ensure pes.potential() handles inputs safely and returns a sensible value

        traj[i] = np.array([p[0], p[1], E])
        integrator.step(step)
    
    traj_all = traj #np.vstack(traj_all_save)
    # success identification of the cluster
    # count number of data points within x_limit and y_limit
    data_count = len(traj_all[(traj_all[:,0] > x_limits[0]) & (traj_all[:,0] < x_limits[1]) \
                        & (traj_all[:,1] > y_limits[0]) & (traj_all[:,1] < y_limits[1])])
    if data_count > num_step/2:
        mean_x = np.mean(traj_all[:,0])
        num_identified_cluster = success_cluster_identification(traj_all[:,:2], 
                                                           center_points, 
                                                           distance_threshold=0.1, 
                                                           number_threshold=5)
        total_num_success += num_identified_cluster
        total_num_clusters += len(center_points)
        init = np.append(init, np.array([[num_identified_cluster]]), axis=1)
        valid_init.append(init)
        plot_save_file = f'{init_no}_{num_identified_cluster}.png'
    else:
        plot_save_file = f'{init_no}_0_fail.png'
    # save plot
    plt.close()
    pes.plot(ax=plt.gca(), 
         minx=x_limits[0]-e, maxx=x_limits[1]+e, 
         miny=y_limits[0]-e, maxy=y_limits[1]+e,
         fontsize=ft-4)
    
    colors = cm.inferno(np.linspace(0.8, 0.2))
    plt.plot(init[:, 0], init[:, 1], '*', 
             markeredgecolor='white', color='r', 
             markersize=10, label='Initial Point')
    plt.scatter(traj_all[:,0], traj_all[:,1], 
            edgecolor='black', 
            s=8, color=colors[0], 
            alpha=0.5) #, label=f'Data Point')
    
    plt.xlabel('$x_1$', fontsize=ft)
    plt.ylabel('$x_2$', fontsize=ft)
    plt.xlim(x_limits[0], x_limits[1])
    plt.ylim(y_limits[0], y_limits[1])
    plt.xticks(fontsize=ft-3)
    plt.yticks(fontsize=ft-3)
    plt.plot([], [], '.', markeredgecolor='black', c='orange', markersize=10, label='Data Point') 
    #plt.legend(fontsize=ft-5, loc='lower left', framealpha=0.5)
    plt.savefig(os.path.join(directory, plot_save_file), bbox_inches='tight', facecolor='w')
    plt.close()
    init_no += 1

ssir = total_num_success / total_num_clusters #(len(center_points)*len(initial_points))
# save ssir as txt file
with open(os.path.join(directory, 'ssir.txt'), 'w') as file:
    file.write(str(ssir))
    file.write('\n')

with open(os.path.join(directory, 'valid_init.pkl'), 'wb') as file:  
    pickle.dump(valid_init, file)

# Plot the starting points
pes.plot(ax=plt.gca(), 
         minx=x_limits[0]-e, maxx=x_limits[1]+e, 
         miny=y_limits[0]-e, maxy=y_limits[1]+e,
         fontsize=ft-4)


for x in grid_x:
    plt.axvline(x, color='lightgray', linestyle='--')

# Plot horizontal lines for each y_grid point
for y in grid_y:
    plt.axhline(y, color='lightgray', linestyle='--')

valid_initial_points = np.vstack(valid_init) #np.array(valid_init)

# Define colors for each value
colors = {1: 'red', 2: 'orange', 3: 'green'}

# Loop through each point and plot it with the corresponding color
for point in valid_initial_points:
    plt.plot(point[0], point[1], '.', 
             color=colors[point[3]], markeredgecolor='black', markersize=15)

for i in range(len(colors)):
    plt.plot([], [], '.', color=list(colors.values())[i], 
             markeredgecolor='black', markersize=12, label=f'{i+1} success')
plt.legend(fontsize=ft-5, loc='lower left', framealpha=0.5)

plt.xlabel('$x_1$', fontsize=ft)
plt.ylabel('$x_2$', fontsize=ft)
plt.xlim(x_limits[0], x_limits[1])
plt.ylim(y_limits[0], y_limits[1])
plt.xticks(fontsize=ft-3)
plt.yticks(fontsize=ft-3)
plt.savefig(os.path.join(directory,'grid.png'), bbox_inches='tight', facecolor='w')