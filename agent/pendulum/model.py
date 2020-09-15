import numpy as np
import copy
from scipy.integrate import odeint

from agent.pendulum.config import SIMULATION
from agent.utils import wrap


class Model(object):

    def __init__(self, hyperparams):
        config = copy.deepcopy(SIMULATION)
        config.update(hyperparams)

        self._hyperparams = config

        # constants
        self.dt = config['dt']
        self.dX = config['common']['dX']
        self.dU = config['common']['dU']

        self._g = config['model']['g']
        self._m = config['model']['m']
        self._l = config['model']['l']
        self._b = config['model']['b']

        self._max_torque = config['model']['max_torque']
        # self._max_omega = config['model']['max_omega']

    def step(self, x, u):
        """
        does one integration step with a step size of dt
        :param x:   state
        :param u:   action
        :return:    next state s'
        """
        x = np.squeeze(np.reshape(x, (self.dX, 1)))
        t = np.linspace(0, self.dt, num=2)
        dx = odeint(self.dxdt, x, t, args=(u, ))

        theta, omega = dx[-1]
        # theta = wrap(theta, m=-np.pi, M=np.pi)
        return np.array([theta, omega])

    def dxdt(self, x, t, u):
        u[np.isnan(u)] = 0
        u = np.reshape(u, (self.dU, 1))

        u[u > self._max_torque] = self._max_torque
        u[u < -self._max_torque] = -self._max_torque

        # scale action
        u_ = u * 50
        theta, omega = x

        dx = np.array([
            omega,
            1/(self._m * self._l**2) * (u_ - self._b * omega - self._m * self._g * self._l * np.sin(theta))
        ], dtype=np.float64)

        return dx
