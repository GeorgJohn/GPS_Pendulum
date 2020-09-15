import pandas as pd


class TabularConditioner(object):

    def __init__(self, columns):

        self._pd = pd.DataFrame(columns=columns)

    def append(self, data):
        self._pd = self._pd.append(data, ignore_index=True)

    def get_data_frame(self):
        return self._pd
