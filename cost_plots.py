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

split_dir = os.path.split(os.path.split(data_files_dir)[0])
data_files_dir = os.path.join(split_dir[0], 'data_files_noisy_inputs/')

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

data_path = '/home/gjohn/tmp/gps_plot_data/traj_cost_plots/'


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

    data_list = find_available_files(data_files_dir, 'global_itr_data')

    last_data = data_list[-1]

    df = data_logger.unpickle(last_data)

    avg_pol_cost = list(df['avg_pol_cost'])
    avg_cost = list(df['avg_cost'])

    avg_pol_cost = [avg_cost[0]] + avg_pol_cost[:-1]

    df['avg_cost'] = avg_cost
    df['avg_pol_cost'] = avg_pol_cost

    # df.plot(x='itr')
    #
    # plt.grid(True)
    # plt.show()

    # global cost iteration
    data_list = find_available_files(data_files_dir, 'global_cost_itr')

    last_data = data_list[-1]

    data_dict = data_logger.unpickle(last_data)

    df = pd.DataFrame({
        'avg_cost_all': np.squeeze(data_dict['data_mean']),
        'avg_cost_con1': data_dict['data'][0, :],
        'avg_cost_con2': data_dict['data'][1, :],
        'avg_cost_con3': data_dict['data'][2, :],
        'avg_cost_con4': data_dict['data'][3, :],
    }
    )

    df.index.name = 'iter'

    # create directory
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    file_name = 'avg_traj_cost.csv'

    # save data as csv file
    df.to_csv(os.path.join(data_path, file_name))

    print(df.head())


if __name__ == '__main__':
    main()
