import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import ndimage

__author__ = 'YBeer'

"""
Monte-Carlo simulation of standard deviation of positioning estimations from 4 Access Points using maximum probability.
Each AP have 7.5 degrees normal error in measurement. The script gives the average standard deviation and a heatmap of
the standard-deviation(X, Y)
"""

"""
Constants
"""
# Print human readable numpy arrays
np.set_printoptions(precision=2, suppress=True)

# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
# [0: x, 1: y, 2: direction]
aps_raw = np.array([[-1, -1, 45.0],
                   [-1, 11, 135.0],
                   [11, 11, 225.0],
                   [11, -1, 315.0]])

# Grid dimesions in meters, resolution of 1 meter [[x_min, x_max], [y_min, y_max]]
boundaries = [[0, 10], [0, 10]]

# Creating calculating grid mesh
x = y = np.arange(0, 10 + 1)
X, Y = np.meshgrid(x, y)

# Recieved DOA noise
noise = 0.01

# Grid's resolution
res = 1

# Monte Carlo repetitions
n_rep = 1

"""
Classes and functions
"""


class AccessPoint(object):
    """
    Access point class.
    x: AP's x
    y: AP's y
    heading: AP's 0 local angle
    """
    def __init__(self, ap_params):
        self.x = ap_params[0]
        self.y = ap_params[1]
        self.heading = ap_params[2]


class Grid(object):
    """
    Calculation coarse grid.
    X: X coordinates matrix
    Y: Y coordinates matrix
    resolution: Resolution of the grid in meters
    max: maximum coordinate (x = y)
    min: minimum coordinate
    """
    def __init__(self, mini, maxi, resolution):
        x = y = np.arange(mini, maxi + 1, resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.resolution = resolution
        self.max = maxi
        self.min = mini


class SimulatedDoa(object):
    """
    Simulated DOA class for the full grid
    ap: The access point simulated
    doa: The real doa from the measured point
    doa_noised: The doa estimated from the measured point
    doa_grid: The calculation grid for each doa
    doa_grid_res: The resolution of the calculation grid
    """
    def __init__(self, ap_cls, x_cells, y_cells):
        """
        Get the true DOA for each coordinate
        :param ap_cls: access point's place and heading
        :param x_cells: cell x coordinate
        :param y_cells: cell y coordinate
        """
        self.ap = ap_cls
        dx = x_cells - self.ap.x
        dy = y_cells - self.ap.y
        g_angle = global_angle_calc(dx, dy)

        self.doa = g_angle - self.ap.heading
        self.doa = (self.doa + 180) % 360 - 180
        self.doa_noised = None
        self.doa_grid = None
        self.doa_grid_res = None

    def add_noise(self, noise_level):
        # adding noise into the true DOA
        self.doa_noised = self.doa + np.random.normal(0, noise_level, self.doa.shape)

    def add_grid(self, grid):
        # adding calculating grid with the true DOA
        dx = grid.X - self.ap.x
        dy = grid.Y - self.ap.y
        g_angle = global_angle_calc(dx, dy)
        self.doa_grid = g_angle - self.ap.heading
        self.doa_grid = (self.doa_grid + 180) % 360 - 180
        self.doa_grid_res = grid.resolution


def global_angle_calc(deltax, deltay):
    # Global angle calculation
    # The angles are like a compass, y+ = 0, x+ = 90, clockwise
    # fixing divide by zero
    dy_0 = (deltay == 0) * 1
    y_div_0 = (deltax > 0) * 90 + (deltax < 0) * 270
    dx_0 = (deltax == 0) * 1
    x_div_0 = (deltay > 0) * 0 + (deltay < 0) * 180

    # calculate angles where dy is non zero
    if deltay.shape:
        if np.any(deltay == 0):
            deltay[deltay == 0] = 1
    else:
        if not deltay:
            deltay = 1
    g_angle = np.arctan(deltax / deltay) * 180 / np.pi
    # fix for negative dy
    g_angle += (deltay < 0) * 180
    # fix for negative dx positive dy
    g_angle += ((deltay > 0) * (deltax < 0)) * 360

    g_angle = g_angle * (1 - dy_0) * (1 - dx_0) + y_div_0 * dy_0 + x_div_0 * dx_0
    return g_angle


def prob_density_grid_max(doa_array, sd):
    # Global coarse brute force max probability estimation
    x_est = np.ones(doa_array[0].doa.shape)
    y_est = np.ones(doa_array[0].doa.shape)
    grid_n_rows = doa_array[0].doa_grid.shape[0]
    res = (doa_array[0].doa.shape[0] / doa_array[0].doa_grid.shape[0])

    def relative_angle_calc(angle1, angle2):
        relative_angle1 = np.abs(angle1 - angle2)
        relative_angle2 = 360 - np.abs(angle1 - angle2)
        return np.minimum(relative_angle1, relative_angle2)

    for i_in in range(doa_array[0].doa.shape[0]):
        for j_in in range(doa_array[0].doa.shape[1]):
            prob_mat = np.zeros(doa_array[0].doa_grid.shape)
            for k_in in range(doa_array.shape[0]):
                prob_mat += relative_angle_calc(doa_array[k_in].doa_grid, doa_array[k_in].doa_noised[i_in, j_in]) ** \
                            2 / (2 * sd ** 2)
            dens_pos = np.argmin(prob_mat)
            x_est[i_in, j_in] = (dens_pos % grid_n_rows + 0.5) * res
            y_est[i_in, j_in] = (dens_pos / grid_n_rows + 0.5) * res
    return x_est, y_est


def prob_density_calc_neldermead(doa_array, sd, rough_x, rough_y):
    # local Nelder-Mead max probability estimation
    def relative_angle_calc(angle1, angle2):
        relative_angle1 = np.abs(angle1 - angle2)
        relative_angle2 = 360 - np.abs(angle1 - angle2)
        return np.minimum(relative_angle1, relative_angle2)

    def local_density(location):
        prob = 0
        for k_in in range(doa_array.shape[0]):
            dx = location[0] - doa_array[k_in].ap.x
            dy = location[1] - doa_array[k_in].ap.y
            g_angle = global_angle_calc(dx, dy)
            l_angle = g_angle - doa_array[k_in].ap.heading
            prob += relative_angle_calc(l_angle, doa_array[k_in].doa_noised[i_in, j_in]) ** 2 / (2 * sd ** 2)
        return prob

    x_est = np.ones(doa_array[0].doa.shape)
    y_est = np.ones(doa_array[0].doa.shape)
    for i_in in range(doa_array[0].doa.shape[0]):
        for j_in in range(doa_array[0].doa.shape[1]):
            res = minimize(local_density, [rough_x[i_in, j_in], rough_y[i_in, j_in]], method='Nelder-Mead',
                           options={
                               'maxiter': 8,
                               # 'disp': True
                           }
                           )
            x_est[i_in, j_in] = res.x[0]
            y_est[i_in, j_in] = res.x[1]

    return x_est, y_est


def mat_fun_calc(fun, matrix_list):
    # applying a function over a list of matrixes, cell wise
    answer = np.ones(matrix_list[0].shape)
    for i_in in range(answer.shape[0]):
        for j_in in range(answer.shape[1]):
            cur_pos_est = []
            for k_in in range(len(matrix_list)):
                cur_pos_est.append(matrix_list[k_in][i_in, j_in])
            answer[i_in, j_in] = fun(cur_pos_est)
    return answer

"""
Program starts here
"""
# init APs
print('Initalizing APs')
ap_arr = []
for i in range(aps_raw.shape[0]):
    ap_arr.append(AccessPoint(aps_raw[i, :]))
ap_arr = np.array(ap_arr)

# init DOA
print('Initalizing true DOAs')
doa_arr = []
for i in range(len(ap_arr)):
    doa_arr.append(SimulatedDoa(ap_arr[i], X, Y))
doa_arr = np.array(doa_arr)

# init grid
print('Initalizing coarse grid calculation')
grid = Grid(np.min(boundaries), np.max(boundaries), res)
for i in range(len(ap_arr)):
    doa_arr[i].add_grid(grid)

# monte-carlo
x_est_err_list = y_est_err_list = []
print('Starting Monte-Carlo simulation with %d repetitions' % n_rep)
for repetition in range(n_rep):
    print('Repetition # %d' % repetition)
    # add noise
    for i in range(len(ap_arr)):
        doa_arr[i].add_noise(noise)

    # find position
    # grid search
    x_est, y_est = prob_density_grid_max(doa_arr, noise)
    # fine search
    x_est, y_est = prob_density_calc_neldermead(doa_arr, noise, x_est, y_est)

    x_est_err_list.append(x_est - X)
    y_est_err_list.append(y_est - Y)

# calculate functions
print('Finding standard deviation')
sd_x = mat_fun_calc(np.std, x_est_err_list)
sd_y = mat_fun_calc(np.std, y_est_err_list)
sd_tot = np.sqrt(sd_x ** 2 + sd_y ** 2)

print('The mean SD is: %f' % np.mean(sd_tot))

# sd_tot = ndimage.filters.uniform_filter(sd_tot, size=3)

# plot SD(x, y) plot
# plt.hist(sd_tot.flatten())
# plt.show()
origin = 'lower'
plt.contourf(X, Y, sd_tot, origin=origin, cmap=plt.cm.hot)
plt.colorbar(orientation='vertical', shrink=0.8)
plt.show()
