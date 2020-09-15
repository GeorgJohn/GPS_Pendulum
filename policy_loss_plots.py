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

data_path = '/home/gjohn/tmp/gps_plot_data/policy_loss_plots/'


def find_available_files(dir_, file_name):
    all_files = os.listdir(dir_)
    available_files = []

    for file_path in all_files:
        if file_name in file_path:
            available_files.append(dir_ + file_path)

    available_files.sort()
    return available_files


def main():

    # find out the number of iterations
    n_itr = len(find_available_files(data_files_dir, 'global_itr_data'))

    data_path = find_available_files(data_files_dir, 'policy_train_loss_history')[-1]

    data = data_logger.unpickle(data_path)

    full_loss = []

    for itr_loss in data:
        for loss in itr_loss:
            full_loss.append(loss)

    itr = 50 * np.arange(0, len(full_loss))

    fig, ax = plt.subplots()
    ax.plot(itr, full_loss)
    ax.set(xlabel='train iteration', ylabel='loss',
           title='Policy Loss')

    ax.grid()
    plt.show()


    #
    #
    # df.index.name = 'iter'
    #
    # # create directory
    # if not os.path.exists(data_path):
    #     os.mkdir(data_path)
    #
    # file_name = 'avg_traj_cost.csv'
    #
    # # save data as csv file
    # df.to_csv(os.path.join(data_path, file_name))
    #
    # print(df.head())


if __name__ == '__main__':
    main()
