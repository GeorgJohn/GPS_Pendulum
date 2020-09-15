import numpy as np
from copy import deepcopy

from agent.agent import Agent
from agent.config import AGENT_SIM
from agent.pendulum.config import WORLD
from agent.utils import generate_noise
from proto.gps_pb2 import ACTION
from sample.sample import Sample


class AgentSim(Agent):
    """
    Communication class between algorithm/project and simulation
    """
    def __init__(self, hyperparams):
        config = deepcopy(AGENT_SIM)
        config.update(hyperparams)
        Agent.__init__(self, config)
        self._setup_world(self._hyperparams['world'])

    def _setup_world(self, world):
        """
        Helper method for handling setup of the simulation world.
        :param world: Simulation that should use
        :return:      None
        """
        self.x0 = []
        self._worlds = [world(self._hyperparams, WORLD['condition'][i + 1])
                        for i in range(self._hyperparams['conditions'])]
        for world in self._worlds:
            self.x0.append(world.get_x0())

    def sample(self, policy, condition, verbose=False, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information about the trial.
        :param policy:    Policy to used in the trial.
        :param condition: (int) Which condition setup to run.
        :param verbose:   (boolean) Whether or not to plot the trial (not used here).
        :param save:      (boolean) Whether or not to store the trial into the samples.
        :param noisy:     (boolean) Whether or not to use noise during sampling.
        :return:          new sample with trajectory
        """
        # reset simulation
        self._worlds[condition].reset(noisy)

        # create a new sample
        state = self._worlds[condition].get_state()
        new_sample = self._init_sample(state)

        # set actions and noise
        U = np.zeros([self.T, self.dU])
        U[:, :] = np.nan
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams) * 4.0
        else:
            noise = np.zeros([self.T, self.dU])

        # run simulation
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if t+1 < self.T:
                self._worlds[condition].run(U[t, :])
                state = self._worlds[condition].get_state()
                self._set_sample(new_sample, state, t)

        # Store all actions in the new sample. The last action should be nan
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)

        return new_sample

    def _init_sample(self, state):
        """
        Construct a new sample and fill in the first state.
        :param state: first simulation state
        :return:      None
        """
        sample = Sample(self)
        self._set_sample(sample, state, -1)
        return sample

    @staticmethod
    def _set_sample(sample, state, t):
        """
        fill sample with a new state at time step t+1
        :param sample:
        :param state:
        :param t:
        :return:       None
        """
        for sensor in state.keys():
            sample.set(sensor, np.array(state[sensor]), t=t+1)
