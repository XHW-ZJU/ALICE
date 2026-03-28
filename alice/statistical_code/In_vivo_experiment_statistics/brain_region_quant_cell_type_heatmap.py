import pandas as pd
import numpy as np
from numpy import ndarray
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


def read_excel_data(file_path: str) -> dict:
    """
    :param file_path:
    :return:
    """
    data = pd.read_excel(file_path, sheet_name=None)
    processed_data = dict()
    for _, key in enumerate(data.keys()):
        df = data[key]
        df_no_nan = df[df.brain_region != np.nan]
        df_no_nan['target'] = df_no_nan['target1'] / df['target2'] * 100
        df_no_nan = df_no_nan[df_no_nan.target != np.nan]
        groupby_data = df_no_nan.groupby('brain_region')['target'].mean().reset_index()
        groupby_data = groupby_data.set_index('brain_region')
        processed_data[key] = groupby_data.copy()
    return processed_data

def dict_to_matrix(source_data: dict, row, column) -> ndarray:
    """"
    :param source_data:
    :param row:
    :param column:
    :return: one matrix, shape(row, column)
    """
    res = []
    for _, region in enumerate(column):
        temp = [source_data[key].loc[region, 'target'] for key in row]
        res.append(temp)
    return np.array(res)


if __name__ == '__main__':
    # Prepare the path of the data table
    path = ['data_table_neun.xlsx',
            'data_table_gfap.xlsx',
            'data_table_olig2.xlsx']

    # read the excel data
    dict1 = read_excel_data(path[0])
    dict2 = read_excel_data(path[1])
    dict3 = read_excel_data(path[2])

    # prepare the maxtrix
    x = ['BALB-AAV9', 'BALB-ALICE_N2', 'BALB-ALICE_N6',
         'C57-AAV9', 'C57-ALICE_N2', 'C57-ALICE_N6',
         'FVB-AAV9', 'FVB-ALICE_N2', 'FVB-ALICE_N6']
    y = ['Cerebral Cortex', 'Hippocampus', 'Striatum', 'Thalamus', 'ventral Midbrain', 'Cerebellum']
    m1 = dict_to_matrix(source_data=dict1, row=x, column=y)
    m2 = dict_to_matrix(source_data=dict2, row=x, column=y)
    m3 = dict_to_matrix(source_data=dict3, row=x, column=y)
    m = np.concatenate([m1, m3, m2], axis=0)

    # plot the heatmap
    heatmap_data = pd.DataFrame(m, index=y*3, columns=x)
    clist = [(111 / 255, 158 / 255, 188 / 255), (248 / 255, 230 / 255, 210 / 255),
             (161 / 255, 44 / 255, 109 / 255)]
    cmap = LinearSegmentedColormap.from_list('Rd_Bl_Rd', clist, N=256)
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    heat = ax.imshow(heatmap_data.T, cmap='viridis', vmin=0, vmax=100)
    plt.xticks(range(18), y*3, rotation=45)
    plt.yticks(range(9), x)
    plt.xlabel('neun->olig2->GFAP')
    plt.colorbar(heat)
    plt.tight_layout()

    plt.show()




