import copy
import numpy as np
import pandas as pd

from data_conditioner.mean_conditioner import MeanConditioner
from data_conditioner.tabular_conditioner import TabularConditioner
from data_conditioner.utils import data_frame_wrapper


# import constants
from proto.gps_pb2 import ANGULAR_POSITION, ANGULAR_VELOCITY, ACTION


class DataConditioner(object):

    def __init__(self, hyperparams, data_logger):

        self._hyperparams = hyperparams
        self._data_files_dir = hyperparams['data_files_dir']

        self._logger = data_logger
        if 'train_conditions' in hyperparams:
            self._train_conditions = len(hyperparams['train_conditions'])
            self._test_conditions = len(hyperparams['test_conditions'])
        else:
            self._train_conditions = hyperparams['conditions']
            self._test_conditions = self._train_conditions

        self._global_cost_conditioner = MeanConditioner(self._train_conditions)
        self._global_iteration_data = TabularConditioner(['itr', 'avg_cost', 'avg_pol_cost'])

        self._local_itr_data = [
            TabularConditioner(['itr', 'cost', 'step', 'entropy', 'pol_cost', 'kl_div_i', 'kl_div_f'])
            for _ in range(self._train_conditions)
        ]

    def update(self, itr, algorithm, agent, traj_sample_lists, pol_sample_lists):
        # log data objects
        self._logger.pickle(
            self._data_files_dir + ('algorithm_itr_%02d.pkl' % itr),
            copy.copy(algorithm)
        )
        self._logger.pickle(
            self._data_files_dir + ('traj_sample_itr_%02d.pkl' % itr),
            copy.copy(traj_sample_lists)
        )
        if pol_sample_lists:
            self._logger.pickle(
                self._data_files_dir + ('pol_sample_itr_%02d.pkl' % itr),
                copy.copy(pol_sample_lists)
            )

        costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]

        global_costs = self._global_cost_conditioner.update(costs, t=itr)

        self._logger.pickle(
            self._data_files_dir + ('global_cost_itr_%02d.pkl' % itr),
            global_costs
        )

        glob_itr_data = self._update_global_iteration_data(itr, algorithm, costs, pol_sample_lists)
        self._logger.pickle(
            self._data_files_dir + ('global_itr_data_%02d.pkl' % itr),
            glob_itr_data
        )

        local_itr_data_list = self._update_local_iteration_data(itr, algorithm, costs, pol_sample_lists)
        cond = 0
        for data in local_itr_data_list:
            cond += 1
            df = data.get_data_frame()
            self._logger.pickle(
                self._data_files_dir + ('%02d_local_itr_data_%02d.pkl' % (cond, itr)),
                df
            )

        self._update_trajectory_visualizations(itr, algorithm, agent, traj_sample_lists, pol_sample_lists)

    def _update_global_iteration_data(self, itr, algorithm, costs, pol_sample_lists):
        """
        Update global iteration data information: iteration and average cost
        """

        avg_cost = np.mean(costs)

        idx = range(self._train_conditions)  # algorithm._hyperparams['test_conditions']
        # pol_sample_lists is a list of singletons
        samples = [sl[0] for sl in pol_sample_lists]
        pol_costs = [np.sum(algorithm.cost[idx].eval(s)[0])
                     for s, idx in zip(samples, idx)]
        avg_pol_cost = np.mean(pol_costs)

        data_dict = {
            'itr': itr,
            'avg_cost': avg_cost,
            'avg_pol_cost': avg_pol_cost,
        }

        self._global_iteration_data.append(data=data_dict)
        return self._global_iteration_data.get_data_frame()

    def _update_local_iteration_data(self, itr, algorithm, costs, pol_sample_lists):
        """
        Update local iteration data information:
        iteration, the mean cost over samples,
        step size, linear Gaussian controller entropies,
        and initial/final KL divergences for BADMM.
        """
        idx = range(self._train_conditions)
        # pol_sample_lists is a list of singletons
        samples = [sl[0] for sl in pol_sample_lists]
        pol_costs = [np.sum(algorithm.cost[idx].eval(s)[0])
                     for s, idx in zip(samples, idx)]

        for m in range(algorithm.M):
            cost = costs[m]
            step = np.mean(algorithm.prev[m].step_mult * algorithm.base_kl_step)
            entropy = 2 * np.sum(np.log(np.diagonal(algorithm.prev[m].traj_distr.chol_pol_covar, axis1=1, axis2=2)))

            kl_div_i = algorithm.cur[m].pol_info.init_kl.mean()
            kl_div_f = algorithm.cur[m].pol_info.prev_kl.mean()

            data_dict = {
                'itr': itr,
                'cost': cost,
                'step': step,
                'entropy': entropy,
                'pol_cost': pol_costs[m],
                'kl_div_i': kl_div_i,
                'kl_div_f': kl_div_f,
            }
            self._local_itr_data[m].append(data=data_dict)
        return self._local_itr_data

    def _update_trajectory_visualizations(self, itr, algorithm, agent, traj_sample_lists, pol_sample_lists):
        """
        Update trajectory visualizations information: the trajectory samples,
        policy samples, and linear Gaussian controller means and covariances.
        """
        position = ['theta']
        velocity = ['theta_dot']
        action = ['torque']

        for m in range(self._train_conditions):
            # store conditioned trajectory samples
            traj_pos_sample_array = self._prepare_samples(traj_sample_lists, m, ANGULAR_POSITION)
            traj_pos_df = data_frame_wrapper(traj_pos_sample_array, position)
            self._logger.pickle(
                self._data_files_dir + ('%02d_traj_samples_pos_%02d.pkl' % (m+1, itr)),
                traj_pos_df
            )
            traj_vel_sample_array = self._prepare_samples(traj_sample_lists, m, ANGULAR_VELOCITY)
            traj_vel_df = data_frame_wrapper(traj_vel_sample_array, velocity)
            self._logger.pickle(
                self._data_files_dir + ('%02d_traj_samples_vel_%02d.pkl' % (m + 1, itr)),
                traj_vel_df
            )
            traj_act_sample_array = self._prepare_samples(traj_sample_lists, m, ACTION)
            traj_act_df = data_frame_wrapper(traj_act_sample_array, action)
            self._logger.pickle(
                self._data_files_dir + ('%02d_traj_samples_act_%02d.pkl' % (m+1, itr)),
                traj_act_df
            )
            # x0 = agent.x0[m]
            mu, sigma = algorithm.traj_opt.forward(algorithm.prev[m].traj_distr, algorithm.prev[m].traj_info)
            # mu[:, 0:agent.dX] = mu[:, 0:agent.dX] - x0
            # mu_df = pd.DataFrame.from_records(mu)
            # sigma_df = pd.DataFrame.from_records(sigma)
            self._logger.pickle(
                self._data_files_dir + ('%02d_traj_opt_mu_%02d.pkl' % (m+1, itr)),
                mu
            )
            self._logger.pickle(
                self._data_files_dir + ('%02d_traj_opt_sigma_%02d.pkl' % (m + 1, itr)),
                sigma
            )

            # get GMM prior
            prior = algorithm.cur[m].traj_info.dynamics.get_prior()
            X, U = prior.X, prior.U
            mu_gmm, sigma_gmm = prior.gmm.mu, prior.gmm.sigma
            gmm_data = [X, U, mu_gmm, sigma_gmm]
            self._logger.pickle(
                self._data_files_dir + ('%02d_gmm_prior_%02d.pkl' % (m + 1, itr)),
                gmm_data
            )

            # get system dynamics
            Fm = algorithm.cur[m].traj_info.dynamics.Fm
            fv = algorithm.cur[m].traj_info.dynamics.fv
            dyn_data = [Fm, fv]
            self._logger.pickle(
                self._data_files_dir + ('%02d_dynamics_fit_%02d.pkl' % (m + 1, itr)),
                dyn_data
            )

        for m in range(self._test_conditions):
            # store conditioned policy samples
            pol_pos_sample_array = self._prepare_samples(pol_sample_lists, m, ANGULAR_POSITION)
            pol_pos_df = data_frame_wrapper(pol_pos_sample_array, position)
            self._logger.pickle(
                self._data_files_dir + ('%02d_pol_samples_pos_%02d.pkl' % (m+1, itr)),
                pol_pos_df
            )
            pol_vel_sample_array = self._prepare_samples(pol_sample_lists, m, ANGULAR_VELOCITY)
            pol_vel_df = data_frame_wrapper(pol_vel_sample_array, velocity)
            self._logger.pickle(
                self._data_files_dir + ('%02d_pol_samples_vel_%02d.pkl' % (m + 1, itr)),
                pol_vel_df
            )
            pol_act_sample_array = self._prepare_samples(pol_sample_lists, m, ACTION)
            pol_act_df = data_frame_wrapper(pol_act_sample_array, action)
            self._logger.pickle(
                self._data_files_dir + ('%02d_pol_samples_act_%02d.pkl' % (m+1, itr)),
                pol_act_df
            )

    def _prepare_samples(self, sample_lists, m, sensor):
        """
        Prepare the samples: Reduce the number of samples to one in three to save data and
        save all data in one array (N, T, dX)
        N = Number of samples for condition m
        T = Time horizon divided by 3
        dX = Sensor dimension
        """
        samples = sample_lists[m].get_samples()
        pt_array = None
        for sample in samples:
            pt = sample.get(sensor_name=sensor)  # shape = (T x dX)
            # if sensor == ANGULAR_POSITION:
            #     x0 = pt[0, :]
            #     pt = pt - x0
            if pt_array is None:
                pt_array = pt
            else:
                pt_array = np.hstack([pt_array, pt])
        return pt_array
