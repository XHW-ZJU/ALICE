import pandas as pd
import numpy as np
from numpy import ndarray
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import seaborn as sns


def read_excel_data(file_path: str) -> dict:
    """
    :param file_path:
    :return:
    """
    data = pd.read_excel(file_path, sheet_name=None)
    processed_data = dict()
    for _, key in enumerate(data.keys()):
        df = data[key]
        df_no_nan = df[df.Seq != np.nan]
        df_no_nan['target'] = df_no_nan['RFP'] / df['DAPI'] * 100
        df_no_nan = df_no_nan[df_no_nan.target != np.nan]
        groupby_data = df_no_nan.groupby(['Species', 'Seq'])['target'].mean()
        processed_data[key] = groupby_data.copy()
    return processed_data

def dict_to_matrix(source_data: dict, organs, virus, species) -> ndarray:
    """
    :param species:
    :param virus:
    :param organs:
    :param source_data:
    :return:
    """
    res = []
    for _, key in enumerate(organs):
        temp = []
        for specie in species:
            print('Processing:{}-{}!'.format(key, specie))
            temp += [source_data[key].loc[(specie, i)] for i in virus]
        res.append(temp)
    return np.array(res)


if __name__ == '__main__':
    path = 'pns_quant_heatmap.xlsx'
    dict1 = read_excel_data(path)
    reference = ['liver', 'muscle', 'heart', 'kidney', 'lung', 'spleen', 'drg']
    matrix = dict_to_matrix(dict1, organs=reference,
                            virus=['AAV9', 'ALICE_N2', 'ALICE_N6'],
                            species=['balb', 'C57', 'FVB'])
    heatmap_data = pd.DataFrame(matrix, index=reference,
                                columns=['AAV9_balb', 'ALICE_N2_balb', 'ALICE_N6_balb', 'AAV9_c57', 'ALICE_N2_c57', 'ALICE_N6_c57',
                                         'AAV9_FVB', 'ALICE_N2_FVB', 'ALICE_N6_FVB'])

    keys = ['AAV9_balb', 'ALICE_N2_balb', 'ALICE_N6_balb', 'AAV9_c57', 'ALICE_N2_c57', 'ALICE_N6_c57',
            'AAV9_FVB', 'ALICE_N2_FVB', 'ALICE_N6_FVB']
    for i in range(3):
        temp = heatmap_data[keys[i*3]]
        for j in range(3):
            heatmap_data[keys[i*3+j]] = heatmap_data[keys[i*3+j]]/temp
    clist = [(111 / 255, 158 / 255, 188 / 255), (248 / 255, 230 / 255, 210 / 255),
             (161 / 255, 44 / 255, 109 / 255)]
    fig, ax = plt.subplots()
    cmap = LinearSegmentedColormap.from_list('Rd_Bl_Rd', clist, N=256)
    heat = ax.imshow(heatmap_data.T, cmap='viridis', vmin=0, vmax=7)
    plt.colorbar(heat)
    plt.xticks(range(len(reference)), reference)
    plt.yticks(range(len(keys)), keys)
    plt.tight_layout()

    plt.show()

