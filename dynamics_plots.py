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

plot_data_path = '/home/gjohn/tmp/gps_plot_data/dyn_plots/'


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
        self._idx = 0
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

        # set figure caption
        title_ = f'Fitted Dynamics.  Iteration: {self._idx}'
        self._fig.suptitle(title_, fontsize=16)

        for data in new_data:
            # clear plots
            self._axs[row, col].clear()
            self._axs[row, col].grid(True)
            self._axs[row, col].set_xlim([0, self._T])
            # self._axs[row, col].set_ylim([-0.25, 1.25])

            Fm, fv = data

            x_ = np.zeros(fv.shape)

            xc = np.ones(Fm.shape[-1])

            for t in range(self._T):
                x_[t, :] = Fm[t, :, :]@xc + fv[t, :]

            # plot data
            self._axs[row, col].plot(x_[:, 1], c=mcolors.CSS4_COLORS['crimson'])

            data_dict = {'phi_dot': list(np.squeeze(x_[:, 1]))}

            # store data in a .csv file
            # create data frame to export plot data
            df = pd.DataFrame(data_dict)
            df.index.name = 'idx'

            # create directory
            if not os.path.exists(plot_data_path):
                os.mkdir(plot_data_path)

            # create file name
            file_name = f'cond{m:02d}_iter{self._idx:02d}_dyn_plot.csv'

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
            dyn_files = find_available_files(data_files_dir, ('%02d_dynamics_fit_%02d' % (m, n)))
            if not dyn_files:
                raise ValueError('Data set is not complete!')

            dynamics = data_logger.unpickle(dyn_files[-1])
            for item in dynamics:
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
