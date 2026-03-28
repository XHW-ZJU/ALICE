import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt


def data_to_dict(file_name, sheet_name, mode, *args):
    """
    :param mode: C/T/L
    :param file_name:
    :param sheet_name:
    :return: {AAV9:[Rfp,Dapi], ALICE_N2:[Rfp,Dapi], ALICE_N6:[Rfp,Dapi]}
    """
    data = pd.read_excel(file_name, sheet_name=sheet_name)
    virus_data: Dict[Any, List[Any]] = dict()
    for i in range(data.shape[0]):
        if not (mode == data.loc[i, 'mode']):
            continue
        else:
            # the best case
            if data.loc[i, 'seq'] in virus_data:
                virus_data[data.loc[i, 'seq']].append(data.loc[i, 'RFP'] / data.loc[i, 'DAPI']*100)
            # otherwise, we need initialize
            else:
                temp = [data.loc[i, 'RFP'] / data.loc[i, 'DAPI']*100]
                virus_data[data.loc[i, 'seq']] = temp
    return virus_data


def dict_average(*args):
    """
    :param args:
    :return: dict
    """
    res = []
    for data in args:
        temp = dict()
        for key in data.keys():
            temp[key] = np.average(data[key])
        res.append(temp)
    return res


def push_data(a: dict, *args):
    if not len(args):
        res = []
    else:
        res = args[0]
    for key, array in a.items():
        for value in array:
            temp = [key, value]
            res.append(temp)
            print('raise:{}'.format(temp))
    return res


if __name__ == '__main__':
    pns_file_path = 'spinal_cord_RFP-DAPI_or_RFP_CHAT_barplot.xlsx'
    mice = 'fvb'
    C_ori = data_to_dict(pns_file_path, mice, 'C')
    T_ori = data_to_dict(pns_file_path, mice, 'T')
    L_ori = data_to_dict(pns_file_path, mice, 'L')
    print(T_ori, '\n')

    C, T, L = dict_average(C_ori, T_ori, L_ori)

    name = ['C', 'T', 'L']
    ori = (C_ori, T_ori, L_ori)
    for i, data in enumerate((C, T, L)):
        df_mean = pd.DataFrame.from_dict(data, orient='index')
        df_scatter = pd.DataFrame.from_dict(ori[i], orient='index')
        df_mean['source'] = '{}-{}'.format(mice, name[i])
        save = pd.concat((df_mean, df_scatter), axis=1)

    x = np.array([1, 4.5, 8])
    y1 = []
    y2 = []
    y3 = []
    # C
    fig, ax = plt.subplots()
    virus = ['AAV9', 'ALICE_N2', 'ALICE_N6']
    colors = ['#E7EAE9', '#C094B5', '#B5E0F6']
    ecolors = ['#8F8F8F', '#7A5568', '#3288AC']
    pcolors = ['#C2C2C2', '#956281', '#33AEDB']
    bar = dict()
    stderr = dict()
    scatter = dict()
    for _, seq_ in enumerate(virus):
        bar[seq_] = [data[seq_] for data in [C, T, L]]
        stderr[seq_] = [np.std(data[seq_]) for data in [C_ori, T_ori, L_ori]]
        scatter[seq_] = [data[seq_] for data in [C_ori, T_ori, L_ori]]
    for i, key in enumerate(bar.keys()):
        ax.bar(x, bar[key], width=1, color=colors[i], label=virus[i])
        plt.errorbar(x, bar[key], xerr=None, yerr=stderr[key], fmt='none', ecolor=ecolors[i], elinewidth=1,
                     capsize=2, ms=0, zorder=20, capthick=1)
        x += 1

    # scatter
    bias = 0.2
    for i, seq_ in enumerate(virus):
        start = i+1-0.5
        for j, data in enumerate(scatter[seq_]):
            x = np.linspace(start+bias, start+1-bias, len(data))
            ax.scatter(x, data, c=pcolors[i], zorder=10,
                       edgecolor=ecolors[i], alpha=0.7, s=40)
            start += 3.5

    # anova
    above = 4
    for i, section in enumerate(['C', 'T', 'L']):
        start = 3.5*i + 2
        group1 = scatter['AAV9'][i]
        for j, v in enumerate(virus[1:]):
            group2 = scatter[v][i]
            _, p_val = ttest_ind(group1, group2, nan_policy='omit')
            print(p_val)
            if p_val < 0.001:
                ax.text(start, np.average(group2) + above, '***', ha='center', va='bottom', fontsize=12)
            elif p_val < 0.01:
                ax.text(start, np.average(group2) + above, '**', ha='center', va='bottom', fontsize=12)
            elif p_val < 0.05:
                ax.text(start, np.average(group2) + above, '*', ha='center', va='bottom', fontsize=12)
            start += 1
    plt.xticks(np.array([2, 5.5, 9]), ['Cervical', 'Thoracic', 'Lumbar'])
    plt.ylabel('chat-Chat+ brain cells (%)')
    plt.title('mouse strain:{}'.format(mice))
    plt.ylim(0, 30)

    plt.show()




