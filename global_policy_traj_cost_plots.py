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
data_files_dir_comp1 = os.path.join(split_dir[0], 'data_files_best/')
data_files_dir_comp2 = os.path.join(split_dir[0], 'data_files_best_low_noise/')
data_files_dir_comp3 = os.path.join(split_dir[0], 'data_files_best_very_low_noise/')
# data_files_dir_comp2 = os.path.join(split_dir[0], 'data_files_kl_3/')
# data_files_dir_comp3 = os.path.join(split_dir[0], 'data_files_kl_5/')

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

data_path = '/home/gjohn/tmp/gps_plot_data/traj_pol_cost_plots/'


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

    data_list1 = find_available_files(data_files_dir_comp1, 'global_itr_data')
    data_list2 = find_available_files(data_files_dir_comp2, 'global_itr_data')
    data_list3 = find_available_files(data_files_dir_comp3, 'global_itr_data')

    last_data1 = data_list1[-1]
    last_data2 = data_list2[-1]
    last_data3 = data_list3[-1]

    df1 = data_logger.unpickle(last_data1)
    df2 = data_logger.unpickle(last_data2)
    df3 = data_logger.unpickle(last_data3)

    avg_pol_cost1 = list(df1['avg_pol_cost'])
    avg_cost1 = list(df1['avg_cost'])
    avg_pol_cost1 = [avg_cost1[0]] + avg_pol_cost1[:-1]

    avg_pol_cost2 = list(df2['avg_pol_cost'])
    avg_cost2 = list(df2['avg_cost'])
    avg_pol_cost2 = [avg_cost2[0]] + avg_pol_cost2[:-1]

    avg_pol_cost3 = list(df3['avg_pol_cost'])
    avg_cost3 = list(df3['avg_cost'])
    avg_pol_cost3 = [avg_cost3[0]] + avg_pol_cost3[:-1]

    df1['avg_cost'] = avg_cost1
    df1['avg_pol_cost'] = avg_pol_cost1

    df2['avg_cost'] = avg_cost2
    df2['avg_pol_cost'] = avg_pol_cost2

    df3['avg_cost'] = avg_cost3
    df3['avg_pol_cost'] = avg_pol_cost3

    df_new = pd.DataFrame(
        {
            # 'itr': df1['itr'],
            # 'avg_traj_cost': np.mean(np.array([avg_cost1, avg_cost2, avg_cost3]), axis=0),
            'avg_traj_cost': avg_cost1,
            # 'avg_cost_2': avg_cost2,
            # 'avg_cost_3': avg_cost3,
            'avg_pol_cost': avg_pol_cost1,
            # 'avg_pol_cost_2': avg_pol_cost2,
            # 'avg_pol_cost_3': avg_pol_cost3,
        }
    )

    # df_new.plot(x='itr')
    #
    # plt.grid(True)
    # plt.show()

    # create directory
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    file_name = 'avg_traj_pol_cost.csv'

    df_new.index.name = 'iter'
    # save data as csv file
    df_new.to_csv(os.path.join(data_path, file_name))

    print(df_new.head())


if __name__ == '__main__':
    main()
