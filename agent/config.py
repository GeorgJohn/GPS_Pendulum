import logging

LOGGER = logging.getLogger(__name__)


# Agent
AGENT = {
    'dH': 0,
    'x0var': 0,
    'smooth_noise': True,
    'smooth_noise_var': 0.1,
    'smooth_noise_renormalize': True,
}

AGENT_SIM = {
    'video_record': False,
    'render': True,
}
