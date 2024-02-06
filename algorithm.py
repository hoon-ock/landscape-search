import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


class ObsBoundReInit():
    '''
    Adam-inspired reinitialization method
    Direction of update is determined by the gradient density
    '''
    def __init__(self, traj, cent_prev, inc_prev, delta, beta, k):
        self.traj = traj 
        self.inc_prev = inc_prev
        self.delta = delta
        self.beta = beta
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

        # Optionally, you can add a background filled contour using ax_contours.contourf:
        # ax_contours.contourf(xx, yy, f, levels=levels, cmap='viridis', alpha=0.5)

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

    # def update_position(self, path, grad, cent_prev):
    #     numer = self.beta * (1 + self.increase(path, cent_prev)/self.k)
    #     denorm = np.sqrt(np.sum(grad**2)+0.0001)
    #     update = -numer/denorm * grad
    #     update_rate = np.sqrt(np.sum(update**2))
    #     return self.last + update, update_rate
    def update_position(self, grad, vn):
        numer = self.beta * (1 + self.increase(vn) / self.k)
        denorm = np.sqrt(np.sum(grad**2)+0.0001)
        update = -numer/denorm * grad
        update_rate = np.sqrt(np.sum(update**2))
        return self.last + update, update_rate
















class ObsBoundReInit_RevisitPrev():
    '''
    Adam-inspired reinitialization method
    Direction of update is determined by the gradient density
    '''
    def __init__(self, traj, cent_prev, delta, beta, k):
        self.traj = traj 
        self.delta = delta
        self.beta = beta
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
        contours = plt.contour(xx,yy, f, levels=levels)
        plt.scatter(self.cent_prev[0], self.cent_prev[1])
        return contours

    def density_gradient(self, point, kde):
        dx = point + np.array([[self.delta],[0]])
        dy = point + np.array([[0],[self.delta]])
        
        f = kde(point)
        f_dx = kde(dx)
        f_dy = kde(dy)

        grad = np.array([(f_dx - f)/self.delta, (f_dy - f)/self.delta])
        return grad 
    

    def get_contour_path(self, target_fraction):
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
    
    def determine_revisit(self, path_list, cent_prev):
        # return >=1 -> no new cluster
        # return 0   -> new cluster
        is_inside = 0
        for path in path_list:
            is_inside += int(path.contains_point(cent_prev))
        return is_inside

    def select_boundary_kde(self, path_list, kde_list, cent_prev):
        # iteration starts from the recent boundary path
        # this can be combined with determine_revisit function
        for i, path in enumerate(path_list[::-1]):
            
            is_inside = path.contains_point(cent_prev)
            if is_inside:
                break
        #idx = len(path_list) - i
        kde = kde_list[::-1][i]
        return path, kde      
    
    def increase(self, vn):
        return vn + 1

    def update_position(self, grad, vn):
        numer = self.beta * (1 + self.increase(vn) / self.k)
        denorm = np.sqrt(np.sum(grad**2)+0.0001)
        update = -numer/denorm * grad
        update_rate = np.sqrt(np.sum(update**2))
        return self.last + update, update_rate