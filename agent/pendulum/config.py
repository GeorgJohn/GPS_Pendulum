import numpy as np


SIMULATION = {
    'common': {
        'dX': 2,
        'dU': 1,
    },

    'model': {
        'g': 9.81,  # gravity
        'm': 1.0,   # mass
        'l': 1.0,   # arm length
        'b': 10.0,   # damping factor

        # 'max_omega': 10000.0,  # max angular velocity
        'max_torque': 1.0,  # max applied torque to the system
    },

    'view': {
        'scale_factor': 200,
        'r_joint': 6,
        'r_land_marker': 5,
        'link_size': 3,
        'screen': (500, 650),
        'margin': 150,
        'controller': {
            'len_rail': 300,
            'len_tick': 8,
            'w_rail': 2,
            'off_rail': 30,
            'radius': 8,
            'C_rail': (67, 115, 108),
        },
        'C_info': (207, 207, 167),
        'C_margin': (27, 39, 43),
        'C_ground': (255, 255, 255),  # (15, 15, 15),
        'C_body': (27, 39, 43),  # (65, 105, 255),
        'C_x0': (47, 255, 47),
        'C_light_x0': (24, 127, 24),
        'C_tgt': (255, 48, 48),
    },
}

WORLD = {
    'condition': {
        1: ([+0.95 * np.pi, 0.0]),
        2: ([-0.95 * np.pi, 0.0]),
        3: ([-0.85 * np.pi, 0.0]),
        4: ([+0.85 * np.pi, 0.0]),
    },
}
