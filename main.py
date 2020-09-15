""" This file defines the main object that runs experiments. """

import logging
import os
import sys
import traceback
import importlib
import random
import numpy as np


from utility.data_logger import DataLogger
from data_conditioner.data_conditioner import DataConditioner
from sample.sample_list import SampleList


# Add gps/python to path so that imports work.
sys.path.append('/'.join(str.split(__file__, '/')[:-1]))


class GPSMain(object):
    """ Main class to run algorithms and experiments. """
    def __init__(self, config, quit_on_end=False):
        """
        Initialize GPSMain
        Args:
            config: Hyperparameters for experiment
            quit_on_end: When true, quit automatically on completion
        """
        self._quit_on_end = quit_on_end
        self._hyperparams = config
        self._conditions = config['common']['conditions']
        if 'train_conditions' in config['common']:
            self._train_idx = config['common']['train_conditions']
            self._test_idx = config['common']['test_conditions']
        else:
            self._train_idx = range(self._conditions)
            config['common']['train_conditions'] = config['common']['conditions']
            self._hyperparams = config
            self._test_idx = self._train_idx

        self._data_files_dir = config['common']['data_files_dir']

        self.agent = config['agent']['type'](config['agent'])
        self.data_logger = DataLogger()

        self.data_conditioner = DataConditioner(config['data_conditioner'], self.data_logger)

        config['algorithm']['agent'] = self.agent
        self.algorithm = config['algorithm']['type'](config['algorithm'])

    def run(self):
        """
        Run training by iteratively sampling and taking an iteration.
        Returns: None
        """

        # empty experiment directory, since algorithm starts at iteration 0
        files_in_directory = os.listdir(self._data_files_dir)
        filtered_files = [file for file in files_in_directory if file.endswith(".pkl")]
        for file in filtered_files:
            path_to_file = os.path.join(self._data_files_dir, file)
            os.remove(path_to_file)

        try:
            for itr in range(self._hyperparams['iterations']):
                for cond in self._train_idx:
                    for i in range(self._hyperparams['num_samples']):
                        self._take_sample(itr, cond, i)

                traj_sample_lists = [
                    self.agent.get_samples(cond, -self._hyperparams['num_samples'])
                    for cond in self._train_idx
                ]

                # Clear agent samples.
                self.agent.clear_samples()

                self._take_iteration(itr, traj_sample_lists)
                pol_sample_lists = self._take_policy_samples()
                self._log_data(itr, traj_sample_lists, pol_sample_lists)

        except Exception as e:
            traceback.print_exception(*sys.exc_info())
        finally:
            self._end()

    def _take_sample(self, itr, cond, i):
        """
        Collect a sample from the agent.
        Args:
            itr: Iteration number.
            cond: Condition number.
            i: Sample number.
        Returns: None
        """
        if self.algorithm._hyperparams['sample_on_policy'] and self.algorithm.iteration_count > 0:
            pol = self.algorithm.policy_opt.policy
        else:
            pol = self.algorithm.cur[cond].traj_distr

        self.agent.sample(pol, cond, verbose=(i < self._hyperparams['verbose_trials']))

    def _take_iteration(self, itr, sample_lists):
        """
        Take an iteration of the algorithm.
        Args:
            itr: Iteration number.
        Returns: None
        """
        self.algorithm.iteration(sample_lists)

    def _take_policy_samples(self):
        """
        Take samples from the policy to see how it's doing.
        Returns: None
        """
        if 'verbose_policy_trials' not in self._hyperparams:
            # AlgorithmTrajOpt
            return None
        verbose = self._hyperparams['verbose_policy_trials']

        pol_samples = [[None] for _ in range(len(self._test_idx))]

        # Since this isn't noisy, just take one sample.
        # TODO: Make this noisy? Add hyperparam?
        for cond in range(len(self._test_idx)):
            pol_samples[cond][0] = self.agent.sample(
                self.algorithm.policy_opt.policy, self._test_idx[cond],
                verbose=verbose, save=False, noisy=False)
        return [SampleList(samples) for samples in pol_samples]

    def _log_data(self, itr, traj_sample_lists, pol_sample_lists=None):
        """
        Log data and algorithm.
        Args:
            itr: Iteration number.
            traj_sample_lists: trajectory samples as SampleList object
            pol_sample_lists: policy samples as SampleList object
        Returns: None
        """

        if 'no_sample_logging' in self._hyperparams['common']:
            return

        self.data_conditioner.update(
            itr=itr,
            algorithm=self.algorithm,
            agent=self.agent,
            traj_sample_lists=traj_sample_lists,
            pol_sample_lists=pol_sample_lists
        )

    def _end(self):
        """ Finish running and exit. """
        pass


def main():
    """ Main function to be run. """

    exp_name = 'pendulum'
    silent = False

    # import __file__ as gps_filepath
    gps_filepath = os.path.abspath(__file__)
    gps_dir = '/'.join(str.split(gps_filepath, '/')[:-1]) + '/'
    exp_dir = gps_dir + 'experiments/' + exp_name + '/'
    sys.path.append(exp_dir)

    if silent:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    hyperparams_file = exp_dir + 'hyperparams.py'
    if not os.path.exists(hyperparams_file):
        sys.exit("Experiment '%s' does not exist.\nDid you create '%s'?" % (exp_name, hyperparams_file))

    hyperparams = importlib.import_module('hyperparams')

    seed = hyperparams.config.get('random_seed', 0)
    random.seed(seed)
    np.random.seed(seed)

    gps = GPSMain(hyperparams.config)

    gps.run()


if __name__ == "__main__":
    main()
