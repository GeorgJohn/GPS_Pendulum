import pandas as pd


def data_frame_wrapper(data_records, col_description):
    n = data_records.shape[1] // len(col_description)
    return pd.DataFrame.from_records(data_records, columns=col_description * n)
