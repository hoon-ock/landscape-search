from openmm.unit import kelvin, picosecond, femtosecond, nanometer, dalton
import openmm as mm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

from energy_landscape import MullerForce, ModifiedMullerForce, landscape
from muller_utils import (ObsBoundReInit, 
                          random_init_point, 
                          random_init_point_with_grid,
                          get_grid_intersection_points,
                          target_init_point, 
                          find_region_change, 
                          deepest_well_escape_frame, 
                          success_cluster_identification)

import os, tqdm, pickle, yaml
from datetime import datetime

# ====================================================================
# Grid parameters

x_limits = (-3.4, 1.7)
y_limits = (-1.4, 3.4)
x_limits_grid = (-3.0, 1.3)
y_limits_grid = (-1.2, 3.2)
grid_interval = (1.5, 1.5)
center_points = np.array([[-0.55, 1.45], [-0.1, 0.45], [0.65, 0.02]])
ft = 15
e = 0.2
# Observation Guided Navigation parameters
random_seed = 1
np.random.seed(random_seed)
total_step = 2000       # total number of steps
large_batch_size = 200  # number of steps for outer loop
small_batch_size = 40   # number of steps for inner loop (scout)
k = 50.0                # update rate parameter (higher k means smaller update)
boundary_frac = 0.75    # threshold to determine bounday based on the fraction of points in the region


# Brownian simulation parameters
## each particle is totally independent, propagating under the same potential
nParticles = 1  

temp_non_dimensional = 750
mass_non_dimensional = 1.0
friction_non_dimensional = 100
timestep_non_dimensional = 10.0
step = 100
temperature = temp_non_dimensional * kelvin
mass = mass_non_dimensional * dalton
friction = friction_non_dimensional / picosecond
timestep = timestep_non_dimensional * femtosecond

# ======================================================================

# Get initial points on the grid
_, grid_x, grid_y = random_init_point_with_grid(x_limits_grid, y_limits_grid, grid_interval=grid_interval)
initial_points = get_grid_intersection_points(grid_x, grid_y)

# Get the current time
current_time = datetime.now()
formatted_time = current_time.strftime('%y%m%d_%H%M')
directory = f"results/muller/ogn/ssir_{formatted_time}"
if not os.path.exists(directory):
    os.makedirs(directory)
print(f"Results will be saved in {directory}")

# Define all other parameters and include them in a dictionary
parameters = {
    'start_mode': 'ssir',
    'random_seed': random_seed,
    'total_step': total_step,
    'large_batch_size': large_batch_size,
    'small_batch_size': small_batch_size,
    'k': k,
    'boundary_frac': boundary_frac,
    'nParticles': nParticles,
    'mass': mass_non_dimensional,
    'temperature': temp_non_dimensional,
    'friction': friction_non_dimensional,
    'timestep': timestep_non_dimensional,
    'step': step,
    'grid_interval_x': grid_interval[0],
    'grid_interval_y': grid_interval[1],
}

# Serialize and save as a YAML file
with open(os.path.join(directory, 'param.yaml'), 'w') as file:
    yaml.dump(parameters, file)

# Brownian simulation set-up
system = mm.System()
pes = landscape('Muller')
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
# breakpoint()
for init in tqdm.tqdm(initial_points):
    # print(f'============ Iteration Number: {init_no} ============')
    # Preparation to run simulations
    traj_all_save = []
    # breakpoint()
    init = init.reshape(1, 3)
    cent = init[:,:2].T 
    v = 0
    startingPositions = init
    #########################
    # Start simulation
    remain_step = total_step
    n=1 # outer loop counter
    while remain_step > 0:
        # Initialization    
        context.setPositions(startingPositions)
        context.setVelocitiesToTemperature(temperature)

        # Outer iteration (Run Browninan Simulation)
        traj = np.zeros((large_batch_size,3))
        for i in range(large_batch_size):
            x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)   
            p = x[0]
            E = pes.potential(x=p[0], y=p[1])
            traj[i] = np.array([p[0], p[1], E])
            integrator.step(step)
        traj_all_save.append(traj)

        # Instantiation
        particle_outer = ObsBoundReInit(traj, cent_prev=cent, inc_prev=v,
                                        delta=0.0001, k=k)

        # Secure boundary & centroid & last point & kernel (density gradient)
        # to determine whether the next centroid is inside the boundary
        boundary_path = particle_outer.get_contour_path(target_fraction=boundary_frac)
        kde_outer = particle_outer.kde()

        cent = particle_outer.cent
        last = particle_outer.last
        remain_step -= large_batch_size # step counter

        # Decision block 
        ## Determine whether the centroid inside any boundary paths 
        ## Start inner loop after the decision
        ## If the centroid is outside all boundary paths, this while loop would be skipped!
        m=0 # inner loop counter
        while (particle_outer.increase(boundary_path, cent) != 0) and (remain_step > 0):
            # Instantiation
            particle_inner = ObsBoundReInit(traj, cent_prev=cent, inc_prev=v,
                                            delta=0.0001, k=k)
    
            # Update reinitialization point
            ## Update rate is amplified by the number of times the centroid is revisited
            density_grad = particle_inner.density_gradient(point=last, kde=kde_outer)
            p_new, upd = particle_inner.update_position(grad=density_grad, vn=v) 
            E_new = pes.potential(*p_new)
            startingPositions = np.vstack((p_new, E_new)).T

            # Re-Initialization
            context.setPositions(startingPositions)
            context.setVelocitiesToTemperature(temperature)       
            # Inner iteration (Run Brownian Simulation)
            traj = np.zeros((small_batch_size,3))
            for i in range(small_batch_size):
                x = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)   
                p = x[0]
                E = pes.potential(x=p[0], y=p[1])
                traj[i] = np.array([p[0], p[1], E]) 
                integrator.step(step)
            traj_all_save.append(traj)

            # Secure centroid & last point & v_n for the next run
            v += 1.0      
            cent = particle_inner.cent
            last = particle_inner.last    
            remain_step -= small_batch_size
            m+=1
            if remain_step < 0:
                print('reached the max step')
                break

        # Reset v_n & init point 
        ## init point as last traj from newly found cluster 
        v=0 # update rate reset 
        p_new = last # last point as the new initial point
        E_new = pes.potential(*p_new)
        startingPositions = np.vstack((p_new, E_new)).T
        n+=1
    


    traj_all = np.vstack(traj_all_save)
    E_mean = np.mean(traj_all[:,2])
     # success identification of the cluster
    # breakpoint()
    if E_mean < 10000:
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
         miny=y_limits[0]-e, maxy=y_limits[1]+e)
    
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
         miny=y_limits[0]-e, maxy=y_limits[1]+e)


for x in grid_x:
    plt.axvline(x, color='lightgray', linestyle='--')

# Plot horizontal lines for each y_grid point
for y in grid_y:
    plt.axhline(y, color='lightgray', linestyle='--')
# breakpoint()

valid_initial_points = np.vstack(valid_init) #np.array(valid_init)

# plt.plot(valid_initial_points[:, 0], 
#          valid_initial_points[:, 1], 
#          '.', color='k', markersize=6)
# Define colors for each value
colors = {1: 'red', 2: 'orange', 3: 'green'}

# Loop through each point and plot it with the corresponding color
for point in valid_initial_points:
    plt.plot(point[0], point[1], '.', 
             color=colors[point[3]], markeredgecolor='black', markersize=15)
    # plt.text(point[0], point[1], f'{point[3]}', color='black', fontsize=9, ha='right', va='bottom')
    #plt.text(point[0], point[1], f'{int(point[3])}', color='black', fontsize=11) #, ha='right', va='bottom')
for i in range(len(colors)):
    plt.plot([], [], '.', color=list(colors.values())[i], 
             markeredgecolor='black', markersize=12, label=f'{i+1} success')
plt.legend(fontsize=ft-5, loc='lower left', framealpha=0.5)

# for center in center_points:
#     plt.plot(center[0], center[1], 'x', color='r', markersize=5)

plt.xlabel('$x_1$', fontsize=ft)
plt.ylabel('$x_2$', fontsize=ft)
plt.xlim(x_limits[0], x_limits[1])
plt.ylim(y_limits[0], y_limits[1])
plt.savefig(os.path.join(directory,'grid.png'), bbox_inches='tight', facecolor='w')