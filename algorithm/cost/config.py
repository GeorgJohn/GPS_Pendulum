""" Default configuration and hyperparameter values for costs. """
import numpy as np

from algorithm.cost.utils import RAMP_CONSTANT


# CostState
COST_STATE = {
    'ramp_option': RAMP_CONSTANT,  # How target cost ramps over time.
    'l1': 0.0,
    'l2': 1.0,
    'alpha': 1e-2,
    'wp_final_multiplier': 1.0,  # Weight multiplier on final time step.
    'data_types': {
        'JointAngle': {
            'target_state': None,  # Target state - must be set.
            'wp': None,  # State weights - must be set.
        },
    },
}


# CostSum
COST_SUM = {
    'costs': [],  # A list of hyperparam dictionaries for each cost.
    'weights': [],  # Weight multipliers for each cost.
}


# CostAction
COST_ACTION = {
    'wu': np.array([]),  # Torque penalties, must be 1 x dU numpy array.
}
