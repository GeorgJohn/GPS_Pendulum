"""
Mean Processor

The Mean Processor convert data to a dictionary with numpy arrays.
With structure

{
    data: "data points"
    data_mean: "mean of the data points"
    ts:   "time step array (x-axis)"
}
"""

import numpy as np


class MeanConditioner:

    def __init__(self, data_len):
        self._t = 0
        self._ts = np.empty((1, 0))
        self._data_len = data_len
        self._data_mean = np.empty((1, 0))
        self._data = np.empty((data_len, 0))

    def update(self, x, t=None):
        """
        Update the plots with new data x. Assumes x is a one-dimensional array.
        """

        x = np.ravel([x])

        if not t:
            t = self._t

        assert x.shape[0] == self._data_len
        t = np.array([t]).reshape((1, 1))
        x = x.reshape((self._data_len, 1))
        mean = np.mean(x).reshape((1, 1))

        self._t += 1
        self._ts = np.append(self._ts, t, axis=1)
        self._data = np.append(self._data, x, axis=1)
        self._data_mean = np.append(self._data_mean, mean, axis=1)

        return dict(data=self._data, data_mean=self._data_mean, ts=self._ts)
