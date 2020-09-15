import os
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
from matplotlib.widgets import Button
from matplotlib.patches import Ellipse

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

plot_data_path = '/home/gjohn/tmp/gps_plot_data/gmm/'


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
        self._fig, self._axs = plt.subplots(2, 2, figsize=(12, 8))
        plt.subplots_adjust(bottom=0.2)
        self._max_idx = len(data) - 1
        self._idx = self._max_idx
        self._data = data
        self._T = T
        self._first_call = True
        self._xi_max = np.ones([1, M])
        self._xi_dot_max = np.ones([1, M])
        self.plot_data()

    def next(self, event):
        self._idx += 1
        if self._idx > self._max_idx:
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
        order = 1
        for data in new_data:
            # clear plots
            self._axs[row, col].clear()
            self._axs[row, col].grid(True)
            self._axs[row, col].set_xlim([-0.25, 1.25])
            self._axs[row, col].set_ylim([-0.25, 1.25])

            X, U, mu, sigma = data

            # plot X and U
            # x = np.mean(X[-3:, :, :], axis=0)
            # u = np.mean(U[-3:, :, :], axis=0)
            # xu = np.c_[x[0:self._T - 1], u[0:self._T - 1]]

            xu = np.c_[X[:, 0:self._T - 1, :], U[:, 0:self._T - 1, :]]

            xi = np.linalg.norm(xu, ord=order, axis=2)  # np.sum(np.abs(xu), axis=2)
            xi_dot = np.linalg.norm(X[:, 1:self._T, :], ord=order, axis=2)  # np.sum(np.abs(X[:, 1:self._T, :]), axis=2)

            if self._first_call:
                xi_max = np.amax(xi)
                if xi_max <= 0.0:
                    xi_max = 1e-6
                xi_dot_max = np.amax(xi_dot)
                if xi_dot_max <= 0.0:
                    xi_dot_max = 1e-6

                self._xi_max = xi_max
                self._xi_dot_max = xi_dot_max


            xi_norm = xi / self._xi_max
            xi_dot_norm = xi_dot / self._xi_dot_max

            # plot mu and sigma of the GMM clusters

            # mu_xi = np.sum(np.abs(mu[:, 0:3]), axis=1)  # l1 norm
            # mu_xi_dot = np.sum(np.abs(mu[:, 3:]), axis=1)  # l1 norm
            mu_xi = np.linalg.norm(mu[:, 0:3], ord=order, axis=1)  # l2 norm
            mu_xi_dot = np.linalg.norm(mu[:, 3:], ord=order, axis=1)  # l2 norm

            # normalize mu
            mu_xi_norm = mu_xi / self._xi_max
            mu_xi_dot_norm = mu_xi_dot / self._xi_dot_max

            # get variance
            sigma11 = np.linalg.norm(sigma[:, 0:3, 0:3], ord=order, axis=(1, 2))
            sigma12 = np.linalg.norm(sigma[:, 0:3, 3:], ord=order, axis=(1, 2))
            sigma21 = np.linalg.norm(sigma[:, 3:, 0:3], ord=order, axis=(1, 2))
            sigma22 = np.linalg.norm(sigma[:, 3:, 3:], ord=order, axis=(1, 2))

            N = sigma.shape[0]
            edges = 100  # number of edges to use to construct each ellipse
            p = np.linspace(0, 2 * np.pi, edges)
            xy_ellipse = np.c_[np.cos(p), np.sin(p)]

            # xy_sigma = np.array([
            #     [sigma11, sigma12],
            #     [sigma21, sigma22]
            # ]).reshape([N, 2, 2])/(np.max([self._xi_max, self._xi_dot_max])*10)

            xy_sigma = np.zeros([N, 2, 2])
            xy_sigma[:, 0, 0] = sigma11 / self._xi_dot_max
            xy_sigma[:, 0, 1] = sigma12 / (0.6 * (self._xi_dot_max + self._xi_max))
            xy_sigma[:, 1, 0] = sigma21 / (0.6 * (self._xi_dot_max + self._xi_max))
            xy_sigma[:, 1, 1] = sigma22 / self._xi_max

            u, s, v = np.linalg.svd(xy_sigma)

            ellipses_dict = {}
            # plot sigma as ellipses
            for n in range(0, N):
                xy = np.repeat(np.array([mu_xi_norm[n], mu_xi_dot_norm[n]]).reshape((1, 2)), edges, axis=0)
                xy[:, 0:2] += np.dot(xy_ellipse, np.dot(np.diag(np.sqrt(s[n, :])), u[n, :, :].T))
                self._axs[row, col].plot(xy[:, 0], xy[:, 1],
                                         c=mcolors.CSS4_COLORS['darkred'],
                                         linestyle='dashed',
                                         linewidth=0.75,
                                         )
                key_name = f'sigma_ellipse{n:02d}'
                ellipses_dict.update({key_name + 'X': np.squeeze(xy[:, 0])})
                ellipses_dict.update({key_name + 'Y': np.squeeze(xy[:, 1])})

            # plot data
            K = xi_norm.shape[0]
            alpha_factor = 10
            data_dict = {}
            train_iter = self._idx
            for k in range(K, 5-1, -5):
                xi_plot = xi_norm[k-5:k, :].flatten()
                xi_dot_plot = xi_dot_norm[k-5:k, :].flatten()
                self._axs[row, col].scatter(xi_plot, xi_dot_plot,
                                            # c=mcolors.CSS4_COLORS['seagreen'],
                                            alpha=0.05*alpha_factor, s=3,
                                            edgecolors='none')

                indices = np.arange(xi_plot.shape[0])
                np.random.shuffle(indices)

                data_dict.update({f'xi_iter{train_iter:02d}': xi_plot[indices][0::5]})
                data_dict.update({f'xi_dot_iter{train_iter:02d}': xi_dot_plot[indices][0::5]})

                train_iter -= 1
                if alpha_factor <= 1.0:
                    alpha_factor = 1.0
                else:
                    alpha_factor -= 1.0

            # plot mu
            mu_dict = {'mu': [], 'mu_dot': []}
            for n in range(0, N):
                mu_plot = mu_xi_norm[n]
                mu_dot_plot = mu_xi_dot_norm[n]
                self._axs[row, col].scatter(mu_plot, mu_dot_plot,
                                            c=mcolors.CSS4_COLORS['crimson'],
                                            alpha=1.0, s=5)
                mu_dict['mu'].append(mu_plot)
                mu_dict['mu_dot'].append(mu_dot_plot)

            # store data in a .csv file
            # create data frame to export plot data
            df_scatter = pd.DataFrame(data_dict)
            df_mu = pd.DataFrame(mu_dict)
            df_sig = pd.DataFrame(ellipses_dict)

            # create directory
            if not os.path.exists(plot_data_path):
                os.mkdir(plot_data_path)

            # create file name
            file_name = f'cond{m:02d}_iter{self._idx:02d}_gmm_scatter_plot.csv'
            file_name_mu = f'cond{m:02d}_iter{self._idx:02d}_gmm_mu_plot.csv'
            file_name_sig = f'cond{m:02d}_iter{self._idx:02d}_gmm_sig_plot.csv'

            # save all data as csv file
            df_scatter.to_csv(os.path.join(plot_data_path, file_name), index=False)
            df_mu.to_csv(os.path.join(plot_data_path, file_name_mu), index=False)
            df_sig.to_csv(os.path.join(plot_data_path, file_name_sig), index=False)

            # count condition
            m += 1

            # increment row and column
            if col > row:
                row += 1
                col = 0
            else:
                col += 1

        self._first_call = False
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
