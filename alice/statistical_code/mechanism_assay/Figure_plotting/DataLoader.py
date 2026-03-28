from typing import Any, List, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import ndarray
from operator import index

from pandas import DataFrame
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy


def read_excel_data(file_path: str,
                    protein_list: list,
                    virus_list: list) -> tuple:
    """
    :param file_path: file_path
    :param protein_list: target protein
    :param virus_list: AAV.ALICE-N2/6, AAV9, PHP.eB, etc.
    :return: [data1, data2, data3]
    """
    df = pd.read_excel(file_path)
    matrix = np.zeros(shape=[7, 4])
    bright = np.copy(matrix)
    for i in range(df.shape[0]):
        temp = df.iloc[i, :]
        row = protein_list.index(temp['proteins'])
        col = virus_list.index(temp['virus'])
        matrix[row, col] = temp['Average_trans_ratio']
        bright[row, col] = temp['Average_brightness']

    return matrix, bright

def groupby_average(groups: list, filepath: str, target) -> list[Any]:
    df = pd.read_excel(filepath)
    df = df[df.virus != np.nan]
    df_group = df.groupby(groups)

    if type(target) == str:
        return df_group['AVERAGE_SIG_AREA'].mean()
    else:
        return [df_group[tar].mean().reset_index() for tar in target]







