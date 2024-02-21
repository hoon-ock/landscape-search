import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import numpy as np
import os, pickle
from energy_landscape import landscape
from matplotlib.animation import FuncAnimation
from IPython.display import Image


# set up the potential energy landscape
pes = landscape('Modified_Muller')

# settings for the plot
x_limits = (-3.5, 2.0)
y_limits = (-1.2, 3.5)
ft = 18
e = 0.2
label_1 = 'Langevin Dynamics'
label_2 = 'GradNav'

directory = "results/mod_muller/base/target_240216_2112"
directory2 = "results/mod_muller/ogn/target_240216_2116"

with open(os.path.join(directory, 'result.pkl'), 'rb') as f:
    result = pickle.load(f)

with open(os.path.join(directory2, 'result.pkl'), 'rb') as f:
    result2 = pickle.load(f)

traj_all_save = result['traj_all_save']
traj_all_stacked = np.vstack(traj_all_save)
init = traj_all_stacked[0].reshape(1, -1)

traj_all_save2 = result2['traj_all_save']
traj_all_stacked2 = np.vstack(traj_all_save2)
init2 = traj_all_stacked2[0].reshape(1, -1)

fig, ax = plt.subplots()

# Plot the potential energy surface with pes.plot()
pes.plot(ax=ax,
         minx=x_limits[0]-e, 
         maxx=x_limits[1]+e, 
         miny=y_limits[0]-e, 
         maxy=y_limits[1]+e)

# Initialize scatter plots for the two trajectories
scat = ax.scatter([], [], edgecolor='black', s=8, color='skyblue', alpha=0.5)
scat2 = ax.scatter([], [], edgecolor='black', s=8, color='orange', alpha=0.5)

# Adding dummy plots for legend labels (assuming label_1 and label_2 are defined)
ax.plot([], [], 'o', color='skyblue', markeredgecolor='black',label=label_1)
ax.plot([], [], 'o', color='orange', markeredgecolor='black', label=label_2)

ax.set_xlabel('$x_1$', fontsize=ft)
ax.set_ylabel('$x_2$', fontsize=ft)
ax.set_xlim(x_limits[0], x_limits[1])
ax.set_ylim(y_limits[0], y_limits[1])
ax.legend(loc='lower left', framealpha=0.5)

def init():
    scat.set_offsets(np.empty((0, 2)))
    scat2.set_offsets(np.empty((0, 2)))
    return scat, scat2

def update(frame):
    xy = traj_all_stacked[:frame, :2]
    xy2 = traj_all_stacked2[:frame, :2]
    scat.set_offsets(xy)
    scat2.set_offsets(xy2)  # Fixed to use xy2 for the second trajectory
    return scat, scat2

# Update interval to 50 milliseconds for better visibility and performance 
# min(len(traj_all_stacked), len(traj_all_stacked2))
ani = FuncAnimation(fig, update, 
                    frames=min(len(traj_all_stacked), len(traj_all_stacked2)), 
                    init_func=init, blit=True, interval=1E-6)

# To display in Jupyter Notebook, use the following magic command at the top of your notebook
# %matplotlib notebook

#ani.save(os.path.join(directory2, 'traj_animation_combined.gif'))  # Ensure imagemagick is installed and configured

FFwriter = animation.FFMpegWriter(fps=500)
ani.save(os.path.join(directory2, 'traj_animation_combined.mp4'), 
         writer = FFwriter)