import copy
import numpy as np

from agent.pendulum.config import WORLD
from agent.pendulum.view import View
from agent.pendulum.model import Model
from agent.utils import wrap
from proto.gps_pb2 import ANGULAR_POSITION, ANGULAR_VELOCITY


class SimView(object):
    class __SimView:
        def __init__(self, hyperparams):
            self.view = View(hyperparams)

    view = None

    def __init__(self, hyperparams):
        if not SimView.view:
            SimView.view = SimView.__SimView(hyperparams)

    def _get_view(self):
        return self.view.view


class World(SimView, Model):

    def __init__(self, hyperparams, condition):
        config = copy.deepcopy(WORLD)
        config.update(hyperparams)
        self._hyperparams = config

        SimView.__init__(self, self._hyperparams)
        Model.__init__(self, self._hyperparams)

        self._view = self._get_view()
        # transform condition
        condition[0] += np.pi
        self._view.register_condition(condition)
        self._condition = condition

        self._state = np.array(condition)
        self._xT = np.array([np.pi, 0.0])

    def reset(self, noisy=False):
        if noisy:
            noise = 0.05 * np.random.randn(len(self._condition))
        else:
            noise = 0.0 * np.random.randn(len(self._condition))

        self._state = np.array(self._condition) + noise
        self._render(action=np.array([0.0]))

    def run(self, action):
        self._state = self.step(self._state, action)
        self._render(action)

    def get_state(self):
        """
        Retrieves the state of the simulation
        """
        theta, omega = self._state
        theta = wrap(theta + np.pi, m=-2*np.pi, M=2*np.pi)
        # theta += np.pi

        state = {
            ANGULAR_POSITION: theta,
            ANGULAR_VELOCITY: omega
        }
        return state

    def _render(self, action):
        if self._hyperparams['render']:
            self._view.display_all(self._state, action, self._condition)

    def get_x0(self):
        return np.array(self._condition)
