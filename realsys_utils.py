import numpy as np
import pandas as pd
import random, os, math, tqdm, pickle
import scipy.stats as st

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


class ObsBoundReInit_Algorithm():
    '''
    Adam-inspired reinitialization method
    Direction of update is determined by the gradient density of observations
    '''
    def __init__(self, traj_dict, init_traj_index, beta, k, large_batch, small_batch, iter_frame, contour_frac, save_path, threshold=0.02):
        self.traj_dict = traj_dict # {trajectory index: trajectory} trajectory in a shape of (n_frames, 2)
        #self.all_traj = all_traj # 2d array 
        self.init_point = traj_dict[init_traj_index][0] # init_point
        self.total_frame = iter_frame
        self.iter_frame = iter_frame 
        self.beta = beta
        self.k = k 
        self.v = 0
        self.large_batch = large_batch # number of frames for outer loop simulation
        self.small_batch = small_batch # number of frames for inner loop simulation (scout run)
        self.contour_frac = contour_frac # a fraction to obtain a specific contour
        self.threshold = threshold
        self.init = self.init_point
        self.last = self.init_point #np.empty(2)
        self.cent = np.empty(2)
        #self.collected_trajs = np.empty([self.large_batch, 2]) # change the name to collected_outer_trajs
        self.collected_outer_trajs = np.empty([self.large_batch, 2])
        self.collected_all_trajs = []
        # self.update_rates = []
        self.kde = None 
        self.paths = None
        self.outer_results = {}
        self.inner_results = {}
        self.iteration_counter = {} # {outer_loop_index: number of inner loop iterations}
        self.save_path = save_path 

    def run_simulation(self):
        print('simulation initialization')
        i = 1
        while self.iter_frame > 0:
            self.run_large_batch()
            self.iter_frame -= self.large_batch #len(self.collected_trajs)
            dist_btw_actual_and_proposed = np.sqrt(np.sum((self.init - self.collected_outer_trajs[0])**2))
            self.outer_results.update({i: [self.init, self.collected_outer_trajs[0], dist_btw_actual_and_proposed]})
            self.collected_all_trajs.append(self.collected_outer_trajs)
            print(f'{self.iter_frame} frames remain!')
            # run inner loop 
            j = 0
            while self.centroid_is_inside_contour():
                prev_last = self.last
                proposed_init, update_rate = self.update_position()
                scout_traj = self.get_trajectory(proposed_init, self.small_batch)
                self.init = scout_traj[0]
                self.last = scout_traj[-1]
                self.cent = np.mean(scout_traj[1:], axis=0)
                self.iter_frame -= len(scout_traj)
                self.inner_results.update({(i, j): [proposed_init, self.init, prev_last, update_rate]}) 
                # [proposed init for current run, actual init point of current run, last point of previous run, update rate]
                self.collected_all_trajs.append(scout_traj)
                j+=1

            print(f'cluster transition {i}!')
            self.v = 0
            self.init = self.last # proposed init
            self.iteration_counter.update({i: j})
            i+=1

        print('simulation termination')
        self.save_results()

    def run_large_batch(self):
        self.collected_outer_trajs = self.get_trajectory(self.init, self.large_batch) # self.init
        # update kde, centroid, last point, contours
        self.kde = st.gaussian_kde(self.collected_outer_trajs.T)
        self.cent = np.mean(self.collected_outer_trajs, axis=0)
        self.last = self.collected_outer_trajs[-1]
        self.paths = self.get_contour_path()

    def get_trajectory(self, init_point, n_frames):
        x, y = init_point
        candidate_trajs = []
        global_closest_dist = float('inf')
        global_closest_traj = None
        global_closest_index = None

        for idx, traj in self.traj_dict.items():
            for i, point in enumerate(traj):
                dist = ((point[0] - x)**2 + (point[1] - y)**2)**0.5
                if dist < self.threshold:
                    if len(traj[i:]) >= n_frames:
                        candidate_trajs.append(traj[i:i+n_frames])
                        break
                elif dist < global_closest_dist:
                    global_closest_dist = dist
                    global_closest_traj = traj
                    global_closest_index = i

        # Only append the globally closest trajectory if no trajectories meet the threshold criteria            
        if (not candidate_trajs) and (global_closest_traj is not None) and (len(global_closest_traj[global_closest_index:]) >= n_frames):
            candidate_trajs.append(global_closest_traj[global_closest_index:global_closest_index+n_frames])
        return random.choice(candidate_trajs) if candidate_trajs else None
    
    def gradient_at_point(self):
        x, y = self.last
        # Calculate the partial derivatives of the kde with respect to x and y
        partial_x = self.kde([x + 1e-8, y])[0] - self.kde([x - 1e-8, y])[0]
        partial_y = self.kde([x, y + 1e-8])[0] - self.kde([x, y - 1e-8])[0]
        # Calculate the gradient vector
        gradient_vector = np.array([partial_x, partial_y])
        # Normalize the gradient vector to obtain the gradient direction
        gradient_direction = gradient_vector / np.linalg.norm(gradient_vector)
        return gradient_direction


    def get_contour(self):
        x, y = self.collected_outer_trajs[:,0], self.collected_outer_trajs[:,1]
        # Define the borders
        deltaX = (max(x) - min(x))/4
        deltaY = (max(y) - min(y))/4
        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        pos = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(self.kde(pos).T, xx.shape)
        levels = np.linspace(min(f.flatten()), max(f.flatten()), 10)
        # find the contour lines
        contours = plt.contour(xx,yy, f, levels=levels)
        plt.clf()
        return contours

    def get_contour_path(self):
        contours = self.get_contour()
        # Count the fraction inside each contour line
        n_points = len(self.collected_outer_trajs)
        x, y = self.collected_outer_trajs[:, 0], self.collected_outer_trajs[:, 1]
        for contour in contours.collections:
            paths = contour.get_paths()
            points_inside_contour = 0
            for path in paths:
                points_inside_contour += sum(path.contains_points(np.stack([x, y]).T))
                # Choose the contour line with a number of points within a certain fraction of the total number of points
            if (points_inside_contour > 0) and (points_inside_contour <= n_points * self.contour_frac):
                break
        # Return the path(s) of the chosen contour line(s)
        return paths

    def centroid_is_inside_contour(self):
        x, y = self.cent
        return any([path.contains_point([x, y]) for path in self.paths])

    def update_position(self):
        self.v += 1
        # linear increase
        ## this part can be various functions, e.g. exponential, logaritmic, etc.
        update_rule = -self.beta*(1 + self.v/self.k) 
        update = update_rule * self.gradient_at_point()
        update_rate = np.sqrt(np.sum(update**2)) # update distance
        return self.last + update, update_rate
    
    def save_results(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        results = {'all_traj': self.collected_all_trajs, 
                   'outer': self.outer_results, 
                   'inner': self.inner_results, 
                   'iteration_counter': self.iteration_counter}
        with open(os.path.join(self.save_path, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)



def combine_trajectories(dict1, dict2):
    """
    Combine trajectories from two dictionaries into one, with each value having the shape (10000, 2).
    
    Parameters:
    - dict1: First dictionary with trajectory indices as keys and (10000, 1) arrays as values.
    - dict2: Second dictionary with the same keys as dict1 and (10000, 1) arrays as values.
    
    Returns:
    - A new dictionary with the same keys and combined arrays of shape (10000, 2) as values.
    """
    combined_dict = {}
    for key in dict1:
        if key in dict2:  # Ensure the key exists in both dictionaries
            # Combine the two arrays horizontally to form a new array of shape (10000, 2)
            combined_array = np.hstack((dict1[key], dict2[key]))
            combined_dict[key] = combined_array
    return combined_dict


def iteration_counter_converter(dictionary):
    # Initialize an empty list to store the result
    result = []
    # Initialize a variable to keep track of the cumulative sum of values
    idx =  0 
    for key, value in dictionary.items():
        result.append((idx, idx+value))
        idx = idx+value+1
    return result


def density_contour(traj, xlim=[1.25, 2.95], ylim=[1.25, 2.95], xlabel='A9 O-C-N (rad)', ylabel='R20 O-CA-CB (rad)'):
    '''Plot the density contours''' 
    #data = np.concatenate(list(self.traj_dict.values()))
    x = traj[:, 0]
    y = traj[:, 1]
    xmin, xmax = np.min(x)-0.1, np.max(x)+0.15
    ymin, ymax = np.min(y)-0.1, np.max(y)+0.15
    xmargin = (xmax-xmin)/20
    ymargin = (ymax-ymin)/20
    minx, maxx = xmin-xmargin, xmax+xmargin
    miny, maxy = ymin-ymargin, ymax+ymargin
    grid_width = max(maxx-minx, maxy-miny) / 200.0
    xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
    pos = np.vstack([xx.ravel(), yy.ravel()])
    kernel = np.vstack([x, y])
    kde = st.gaussian_kde(kernel)
    density = kde(pos).reshape(xx.shape)
    CS = plt.contourf(xx, yy, density, 60, cmap=plt.cm.viridis.reversed())
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    cbar = plt.colorbar(CS)
    cbar.ax.set_ylabel('Density')


def get_data_for_figures(inner_results):
    proposed_inits, new_inits, prev_lasts, updates = [], [], [], []
    for value in inner_results.values():
        # proposed_init, new_init, self.last, update_rate
        pro_init, new_init, prev_last, update = value
        proposed_inits.append(pro_init)
        new_inits.append(new_init)
        prev_lasts.append(prev_last)
        updates.append(update)
        if not math.isclose(math.dist(pro_init, prev_last), update, rel_tol=0.001):
            print('update unmatching!', math.dist(pro_init, prev_last), update)
    return np.stack(proposed_inits), np.stack(new_inits), np.stack(prev_lasts), updates

def reconstruct_energy(data, bin_number=500):
    counts, bins = np.histogram(data, bins=bin_number)

    anchors = (bins[1:] + bins[:-1]) / 2

    probs = counts / np.sum(counts)

    anchors = anchors[np.where(probs > 0.0001)]
    probs = probs[np.where(probs > 0.0001)]

    f = -np.log(probs)
    fn = f - np.min(f)
    return anchors, fn 