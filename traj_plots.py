import os
import sys
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.markers as mmarkers
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

plot_data_path = '/home/gjohn/tmp/gps_plot_data/traj_plots/'


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
        self._idx = 0
        self._max_idx = len(data) - 1
        self._data = data
        self._T = T
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

        # set figure caption
        title_ = f'Trajectories    Iteration: {self._idx}'
        self._fig.suptitle(title_, fontsize=16)

        for data in new_data:
            # clear plots
            self._axs[row, col].clear()
            self._axs[row, col].grid(True)
            self._axs[row, col].set_xlim([0, self._T])
            self._axs[row, col].set_ylim([- 4, + 4])

            # plot target
            self._axs[row, col].plot([0, self._T], [0, 0], mcolors.CSS4_COLORS['red'])

            self._axs[row, col].plot(
                [0, self._T],
                [condition[m][0], condition[m][0]],
                mcolors.CSS4_COLORS['forestgreen'],
            )

            # plot mu and sigma of learned trajectory
            mu = data[2]
            sigma = data[3]
            theta_sig = 1*sigma[:, 0, 0]

            # plot mu as graph
            self._axs[row, col].plot(mu[:, 0], mcolors.CSS4_COLORS['mediumorchid'], label='Fw-pass')

            data_dict = {'mu': mu[:, 0]}

            xy = np.zeros([2, 2])
            # plot sigma as bar
            sig_dict = {'sigma_min': [], 'sigma_max': [], 'sigma_error': []}
            for t in range(T):
                y = mu[t, 0]
                xy[:, 0] = t
                xy[0, 1] = y + theta_sig[t]
                xy[1, 1] = y - theta_sig[t]
                if t % 3 == 0:
                    self._axs[row, col].plot(xy[:, 0], xy[:, 1], mcolors.CSS4_COLORS['darkorchid'],
                                             marker='_', markersize=3)

                sig_dict['sigma_min'].append(xy[1, 1])
                sig_dict['sigma_max'].append(xy[0, 1])
                sig_dict['sigma_error'].append(theta_sig[t])

            data_dict.update(sig_dict)

            # plot trajectories created by parameterized policy
            self._axs[row, col].plot(data[1][:], mcolors.CSS4_COLORS['navy'], label='policy')
            data_dict.update({'pol': list(np.squeeze(data[1][:]))})

            # plot trajectories created by linear Gaussian controller
            self._axs[row, col].plot(data[0][:], mcolors.CSS4_COLORS['tomato'], label='traj. opt')
            data_dict.update({'traj': list(np.squeeze(data[0][:]))})

            # store data in a .csv file
            # create data frame to export plot data
            df = pd.DataFrame(data_dict)
            df.index.name = 'idx'

            # create directory
            if not os.path.exists(plot_data_path):
                os.mkdir(plot_data_path)

            # create file name
            cond_deg = int(condition[m][0] * 180 / np.pi)
            file_name = f'cond{m:02d}_deg{cond_deg:03d}_iter{self._idx:02d}_traj_plot.csv'

            # save all data as csv file
            df.to_csv(os.path.join(plot_data_path, file_name))

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
            # read trajectories created by linear Gaussian controller
            traj_file_pos = find_available_files(data_files_dir, ('%02d_traj_samples_pos_%02d' % (m, n)))
            if not traj_file_pos:
                raise ValueError('Data set is not complete!')
            traj_sample = data_logger.unpickle(traj_file_pos[-1])
            traj_mean_theta = np.array([traj_sample.theta.mean(axis=1)]).T
            data[n][m-1].append(traj_mean_theta)

            # read trajectories created by parameterized policy
            pol_file_pos = find_available_files(data_files_dir, ('%02d_pol_samples_pos_%02d' % (m, n)))
            if not pol_file_pos:
                raise ValueError('Data set is not complete!')
            pol_sample = data_logger.unpickle(pol_file_pos[-1])
            pol_pos = np.array([pol_sample.theta]).T
            data[n][m-1].append(pol_pos)

            # read state-action margin created by linear Gaussian controller
            traj_file_mu = find_available_files(data_files_dir, ('%02d_traj_opt_mu_%02d' % (m, n)))
            if not traj_file_mu:
                raise ValueError('Data set is not complete!')
            traj_mu = data_logger.unpickle(traj_file_mu[-1])
            mu = traj_mu[:, 0:2]
            data[n][m - 1].append(mu)

            traj_file_sigma = find_available_files(data_files_dir, ('%02d_traj_opt_sigma_%02d' % (m, n)))
            if not traj_file_sigma:
                raise ValueError('Data set is not complete!')
            traj_sigma = data_logger.unpickle(traj_file_sigma[-1])
            sigma = traj_sigma[:, 0:2, 0:2]
            data[n][m - 1].append(sigma)

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
