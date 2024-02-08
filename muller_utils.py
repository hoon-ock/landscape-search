import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

class ObsBoundReInit():
    '''
    Adam-inspired reinitialization method
    Direction of update is determined by the gradient density
    '''
    def __init__(self, traj, cent_prev, inc_prev, delta, k):
        self.traj = traj 
        self.inc_prev = inc_prev
        self.delta = delta
        self.k = k 
        self.x = traj[1:,0] # not considering init point
        self.y = traj[1:,1] # not considering init point
        self.cent = np.array([[self.x.mean()], [self.y.mean()]])
        self.cent_prev = cent_prev
        self.init = np.array([[traj[0,0]], [traj[0,1]]]) 
        self.last = np.array([[self.x[-1]], [self.y[-1]]]) 

    
    def kde(self):
        x, y = self.x, self.y
        values = np.vstack([x, y])
        kernel = st.gaussian_kde(values)
        return kernel
    
    def contour(self):
        x, y = self.x, self.y
        # Define the borders
        deltaX = (max(x) - min(x))/10
        deltaY = (max(y) - min(y))/10
        xmin = min(x) - deltaX
        xmax = max(x) + deltaX
        ymin = min(y) - deltaY
        ymax = max(y) + deltaY
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        pos = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(self.kde()(pos).T, xx.shape)
        levels = np.linspace(min(f.flatten()), max(f.flatten()), 10)
        # contours = plt.contour(xx,yy, f, levels=levels)

        # Create a separate figure and axis for contours
        fig_contours, ax_contours = plt.subplots()

        # Create the contours on the separate axis
        contours = ax_contours.contour(xx, yy, f, levels=levels)
        plt.close(fig_contours)  # Close the figure to prevent it from being displayed
        # plt.scatter(self.cent_prev[0], self.cent_prev[1])
        return contours

    def density_gradient(self, point, kde):
        dx = point + np.array([[self.delta],[0]])
        dy = point + np.array([[0],[self.delta]])
        
        f = kde(point)
        f_dx = kde(dx)
        f_dy = kde(dy)

        grad = np.array([(f_dx - f)/self.delta, (f_dy - f)/self.delta])
        return grad 
    

    def get_contour_path(self, target_fraction=0.9):
        n_points = len(self.traj)
        x, y = self.x, self.y
        contours = self.contour()
        
        # Initialize the variables to find the contour that contains more than 50% of the data
        contour_index = 0
        # Loop over the contour lines to find the one that contains more than 50% of the data
        for i, contour in enumerate(contours.collections):
            path = contour.get_paths()[0]
            points_inside_contour = sum(path.contains_points(np.stack([x, y]).T))
            if (points_inside_contour > 0) and (points_inside_contour <= n_points * target_fraction):
                contour_index = i
                break
        # Get the coordinates of the contour that contains more than 50% of the data
        path = contours.collections[contour_index].get_paths()[0]
        x_values = path.vertices[:,0]
        y_values = path.vertices[:,1]
        plt.plot(x_values, y_values, color='red', ls='--')
        return path
    
    
    def increase(self, path, cent_prev):
        is_inside = path.contains_point(cent_prev) #path.contains_point(self.cent_prev)
        if is_inside:
            g = 1
        else:
            g = 0
        return g * (self.inc_prev + g)

    def update_position(self, grad, vn):
        numer = 1 + vn / self.k
        denorm = np.sqrt(np.sum(grad**2)+0.0001)
        update = -numer/denorm * grad
        update_rate = np.sqrt(np.sum(update**2))
        return self.last + update, update_rate
    

# initial set up
def random_init_point(x_limits, y_limits, E_limits=(0, 1)):
    """
    Generates a random point within the specified limits for x, y, and E dimensions.

    Parameters:
    - x_limits: Tuple of (min, max) for x dimension.
    - y_limits: Tuple of (min, max) for y dimension.
    - E_limits: Tuple of (min, max) for E dimension, default is (0, 1).

    Returns:
    - A numpy array representing the random point in three dimensions (x, y, E).
    """
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    E_min, E_max = E_limits

    # Generate a random point within the specified limits for x and y
    initial_x = np.random.uniform(x_min, x_max)
    initial_y = np.random.uniform(y_min, y_max)

    # Generate a random value for the third dimension (E)
    initial_E = np.random.uniform(E_min, E_max)

    # Create the initial point as a numpy array in three dimensions (x, y, E)
    initial_point = np.array([initial_x, initial_y, initial_E])

    return initial_point.reshape(1, 3)

def target_init_point(x_target, y_target, std=0.1):
    initial_coordinate = np.array([x_target, y_target, np.random.uniform(0, 1)])
    noise = np.random.normal(0, std, size=3)
    initial_coordinate += noise
    return initial_coordinate.reshape(1, 3)


# metric
def find_region_change(traj_all_save, threshold=0.05):
    # Extract the points from the first trajectory
    first_trajectory = traj_all_save[0][:, :2]

    # Estimate the KDE (Kernel Density Estimation) of the first trajectory
    kde = gaussian_kde(first_trajectory.T)

    required_frames = len(first_trajectory)

    # Iterate through the rest of the trajectories
    for i in range(1, len(traj_all_save)):
        trajectory = traj_all_save[i][:, :2]
        required_frames += len(trajectory)

        # Calculate the centroid of the current trajectory
        centroid_x = np.mean(trajectory[:, 0])
        centroid_y = np.mean(trajectory[:, 1])

        # Calculate the probability density at the centroid
        density_at_centroid = kde([centroid_x, centroid_y])

        # Check if the density at the centroid is non-zero
        if density_at_centroid < threshold:
            break

    return required_frames

def deepest_well_escape_frame(iteration_counter, large_batch_size, small_batch_size):
    return large_batch_size + small_batch_size*(iteration_counter[1]-1)