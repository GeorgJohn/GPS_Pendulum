""" Default configuration for trajectory optimization. """


# TrajOptLQRPython
TRAJ_OPT_LQR = {
    # Dual variable updates for non-PD Q-function.
    'del0': 1e-4,
    'eta_error_threshold': 1e16,
    'min_eta': 1e-8,
    'max_eta': 1e16,
    'cons_per_step': False,  # Whether or not to enforce separate KL constraints at each time step.
    'use_prev_distr': False,  # Whether or not to measure expected KL under the previous traj distr.
    'update_in_bwd_pass': True,  # Whether or not to update the TVLG controller during the bwd pass.
}
