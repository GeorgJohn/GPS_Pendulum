""" Default configuration and hyperparameter values for algorithms. """


# Algorithm
ALG = {
    'inner_iterations': 1,     # Number of iterations.
    'min_eta':          1e-5,  # Minimum initial lagrange multiplier in DGD for trajectory optimization.
    'kl_step':          0.2,
    'min_step_mult':    0.01,
    'max_step_mult':    10.0,
    'min_mult':         0.1,
    'max_mult':         5.0,

    # Trajectory settings.
    'initial_state_var': 1e-6,
    'init_traj_distr':   None,   # A list of initial LinearGaussianPolicy objects for each condition.
    'traj_opt':          None,   # Trajectory optimization.
    'max_ent_traj':      0.0,    # Weight of maximum entropy term in trajectory optimization.
    'dynamics':          None,   # Dynamics hyperparams.
    'cost':              None,   # A list of Cost objects for each condition.
    'sample_on_policy':  False,  # Whether or not to sample with neural net policy (only for BADMM).
    'fit_dynamics':      True,   # Indicates if the algorithm requires fitting of the dynamics.
}


# AlgorithmBADMM
ALG_BADMM = {
    'inner_iterations': 4,
    'policy_dual_rate': 0.1,
    'policy_dual_rate_covar': 0.0,
    'fixed_lg_step': 0,
    'lg_step_schedule': 10.0,
    'ent_reg_schedule': 0.0,
    'init_pol_wt': 0.01,
    'policy_sample_mode': 'add',
    'exp_step_increase': 2.0,
    'exp_step_decrease': 0.5,
    'exp_step_upper': 0.5,
    'exp_step_lower': 1.0,
}
