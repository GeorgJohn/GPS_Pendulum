import os
import sys
import importlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button

from utility.data_logger import DataLogger

# set path and import parameters
exp_name = 'pendulum'

# import __file__ as gps_filepath
gps_filepath = os.path.abspath(__file__)
gps_dir = '/'.join(str.split(gps_filepath, '/')[:-1]) + '/'
exp_dir = gps_dir + 'experiments/' + exp_name + '/'
sys.path.append(exp_dir)

hyperparams_file = exp_dir + 'hyperparams.py'
if not os.path.exists(hyperparams_file):
    sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

hyperparams = importlib.import_module('hyperparams')

data_files_dir = hyperparams.config['common']['data_files_dir']
M = len(hyperparams.config['common']['train_conditions'])

T = hyperparams.config['agent']['T']

data_logger = DataLogger()

condition = {
    1: ([+0.95 * np.pi - 2 * np.pi, 0.0]),
    2: ([-0.95 * np.pi + 2 * np.pi, 0.0]),
    3: ([-0.85 * np.pi + 2 * np.pi, 0.0]),
    4: ([+0.85 * np.pi - 2 * np.pi, 0.0]),
}

target = 0.0


def find_available_files(dir_, file_name):
    all_files = os.listdir(dir_)
    available_files = []

    for file_path in all_files:
        if file_name in file_path:
            available_files.append(dir_ + file_path)

    available_files.sort()
    return available_files


class Plot(object):

    def __init__(self, data, T):
        self._fig, self._axs = plt.subplots(2, 2, figsize=(12, 8), subplot_kw=dict(projection='3d'))
        plt.subplots_adjust(bottom=0.2)
        self._max_idx = len(data) - 1
        self._idx = self._max_idx
        self._data = data
        self._T = T
        self.plot_data()

    def next(self, event):
        self._idx += 1
        if self._idx >= self._max_idx:
            self._idx = 0
        self.plot_data()

    def prev(self, event):
        self._idx -= 1
        if self._idx < 0:
            self._idx = self._max_idx
        self.plot_data()

    def plot_data(self):
        new_data = self._data[self._idx]
        row, col, m = 0, 0, 1
        for data in new_data:
            # clear plots
            self._axs[row, col].clear()
            self._axs[row, col].grid(True)
            # self._axs[row, col].set_xlim([0, self._T])
            # self._axs[row, col].set_ylim([- 4, + 4])

            X, U, mu, sigma = data

            # plot X and U
            theta = X[-12:, 0:self._T - 1, 0]
            theta_dot = X[-12:, 1:self._T, 0]
            torque = np.squeeze(U[-12:, 0:self._T - 1])

            # plot data
            self._axs[row, col].scatter(theta, theta_dot, torque, mcolors.CSS4_COLORS['red'])

            # plot mu and sigma of learned trajectory
            mu_x = mu[:, 3]
            mu_y = mu[:, 0]
            mu_z = mu[:, 2]

            N = sigma.shape[0]
            sigma_xu = np.array([[sigma[:, 3, 3], sigma[:, 3, 2]], [sigma[:, 2, 3], sigma[:, 2, 2]]]).reshape([N, 2, 2])
            edges = 100  # number of edges to use to construct each ellipse
            p = np.linspace(0, 2 * np.pi, edges)
            xy_ellipse = np.c_[np.cos(p), np.sin(p)]

            u, s, v = np.linalg.svd(sigma_xu)

            # plot mu as graph
            self._axs[row, col].scatter(mu_x, mu_y, mu_z, mcolors.CSS4_COLORS['red'], label='Fw-pass')

            # plot sigma as ellipses
            for n in range(0, N):
                xz = np.repeat(np.array([mu_x[n], mu_z[n]]).reshape((1, 2)), edges, axis=0)
                xz[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(np.sqrt(s[n, :])), u[n, :, :].T))
                y = np.repeat(mu_y[n], edges)
                self._axs[row, col].plot(xz[:, 0], y, xz[:, 1], mcolors.CSS4_COLORS['darkred'])

            # count condition
            m += 1

            # increment row and column
            if col > row:
                row += 1
                col = 0
            else:
                col += 1

        plt.draw()


def main():

    # find out the number of iterations
    n_itr = len(find_available_files(data_files_dir, 'global_itr_data'))

    # create data set
    data = []
    for n in range(n_itr):
        data.append([])
        for m in range(1, M + 1):
            data[n].append([])

            # read GMM priors
            gmm_files = find_available_files(data_files_dir, ('%02d_gmm_prior_%02d' % (m, n)))
            if not gmm_files:
                raise ValueError('Data set is not complete!')

            gmm_data = data_logger.unpickle(gmm_files[-1])
            X, U, mu_gmm, sigma_gmm = gmm_data
            for item in gmm_data:
                data[n][m-1].append(item)

    plotter = Plot(data, T)

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(plotter.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(plotter.prev)

    plt.show()


if __name__ == '__main__':
    main()
