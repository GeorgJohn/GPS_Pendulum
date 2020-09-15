import os.path
import numpy as np
from datetime import datetime

from agent.pendulum.agent import AgentSim
from agent.pendulum.world import World

from algorithm.cost.cost_action import CostAction
from algorithm.cost.cost_state import CostState
from algorithm.cost.cost_sum import CostSum

from algorithm.algorithm_badmm import AlgorithmBADMM

from algorithm.policy.lin_gauss_init import init_lqr
from algorithm.policy.policy_prior_gmm import PolicyPriorGMM

from algorithm.traj_opt.traj_opt_lqr import TrajOptLQR

from algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

from algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from algorithm.policy_opt.tf_models import tf_network

# import constants
from proto.gps_pb2 import ANGULAR_POSITION, ANGULAR_VELOCITY, ACTION
from algorithm.cost.utils import RAMP_LINEAR

# set experiments path
EXP_DIR = '/'.join(str.split(__file__, '/')[:-1]) + '/'


SENSOR_DIMS = {
    ANGULAR_POSITION: 1,
    ANGULAR_VELOCITY: 1,
    ACTION: 1
}

CONDITIONS = 4

common = {
    'experiment_name': 'dot2Dsim_example' + '_' + datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': CONDITIONS,
    'train_conditions': range(CONDITIONS),
    'test_conditions': range(CONDITIONS),
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

# Agent
agent = {
    'type': AgentSim,
    'world': World,
    'dt': 0.02,
    'fps': 150,
    'substeps': 1,
    'conditions': common['conditions'],
    'data_files_dir': common['data_files_dir'],
    'T': 150,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [ANGULAR_POSITION, ANGULAR_VELOCITY],
    'obs_include': [ANGULAR_POSITION, ANGULAR_VELOCITY],
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1])
}

state_cost = {
    'type': CostState,
    'ramp_option': RAMP_LINEAR,
    'wp_final_multiplier': 10.0,
    'data_types': {
        ANGULAR_POSITION: {
            'wp': np.array([5]),
            'target_state': np.array([0]),
        },
    },
}

# Algorithm
algorithm = {
    # common
    'type': AlgorithmBADMM,
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'iterations': 12,
    'inner_iterations': 3,
    'sample_on_policy':  False,  # Whether or not to sample with neural net policy.
    'lg_step_schedule': 0.0,  # has no effect since fixed_lg_step == 3
    'policy_dual_rate': 0.001,  # 0.2,
    'ent_reg_schedule': 0.0,  # has no effect since not used in children class of policy_opt.py
    'fixed_lg_step': 3,
    'kl_step': 1.5,  # 5.0,
    'min_step_mult': 0.01,
    'max_step_mult': 1.0,
    'sample_decrease_var': 0.05,
    'sample_increase_var': 0.1,

    # initial trajectory distribution
    'init_traj_distr': {
        'type': init_lqr,
        'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
        'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
        'init_var': 0.005,
        'stiffness': 1.0,
        'dt': agent['dt'],
        'T': agent['T'],
    },

    # policy prior
    'policy_prior': {
        'type': PolicyPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },

    # trajectory optimizer
    'traj_opt': {
        'type': TrajOptLQR,
    },

    # dynamics
    'dynamics': {
        'type': DynamicsLRPrior,
        'regularization': 1e-6,
        'prior': {
            'type': DynamicsPriorGMM,
            'max_clusters': 20,
            'min_samples_per_cluster': 40,
            'max_samples': 20,
        },
    },

    # policy optimizer
    'policy_opt': {
        'type': PolicyOptTf,
        'data_files_dir': common['data_files_dir'],
        'weights_file_prefix': EXP_DIR + 'policy',
        'network_params': {
            'obs_include': [ANGULAR_POSITION, ANGULAR_VELOCITY],
            'sensor_dims': SENSOR_DIMS,
        },
        'iterations': 2500,  # 3000,
        'network_model': tf_network,
    },

    # cost
    'cost': {
        'type': CostSum,
        'costs': [action_cost, state_cost],
        'weights': [0.1, 1.0],
    }
}

conditioner = {
    'conditions': common['conditions'],
    'train_conditions': common['train_conditions'],
    'test_conditions': common['test_conditions'],
    'data_files_dir': common['data_files_dir'],
    'sensor_dims': SENSOR_DIMS,
    'T': agent['T'],
}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 0,
    'verbose_policy_trials': 0,
    'common': common,
    'agent': agent,
    'data_conditioner': conditioner,
    'algorithm': algorithm,
}
