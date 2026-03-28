from pandas import DataFrame
from scipy.stats import ttest_ind
from brain_region_quant_cell_type_heatmap import read_excel_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def dataframe_to_list(df: DataFrame, ref_index: list):
    return [df.loc[index, 'target'] for index in ref_index]

def read_std_err(file_path: str):
    data = pd.read_excel(file_path, sheet_name=None)
    processed_data = dict()
    scatter_data = dict()
    for _, key in enumerate(data.keys()):
        df = data[key]
        df_no_nan = df[df.brain_region != np.nan]
        df_no_nan['target'] = df_no_nan['target1'] / df['target2'] * 100
        df_no_nan = df_no_nan[df_no_nan.target != np.nan]
        std_err = df_no_nan.groupby('brain_region')['target'].sem().reset_index()
        scatter = df_no_nan.groupby('brain_region')['target'].apply(list).reset_index()
        std_err = std_err.set_index('brain_region')
        scatter = scatter.set_index('brain_region')
        processed_data[key] = std_err
        scatter_data[key] = scatter
    return processed_data, scatter_data


# define the path of data table
path = 'brain_region_quant_heatmap.xlsx'
stderr, scatters = read_std_err(path)
ref = ['Cerebral Cortex', 'Hippocampus', 'Striatum', 'Thalamus', 'ventral Midbrain', 'Cerebellum']
data = read_excel_data(path)

for _, key in enumerate(data.keys()):
    data[key] = dataframe_to_list(data[key], ref_index=ref)
    stderr[key] = dataframe_to_list(stderr[key], ref_index=ref)
    scatters[key] = dataframe_to_list(scatters[key], ref_index=ref)

x = ['BALB-AAV9', 'BALB-ALICE_N2', 'BALB-ALICE_N6',
     'C57-AAV9', 'C57-ALICE_N2', 'C57-ALICE_N6',
     'FVB-AAV9', 'FVB-ALICE_N2', 'FVB-ALICE_N6']

# define the mice strains(FVB or C57 or BALB)
mice = 'FVB'
x = [i for i in x if mice in i]
plt.figure(figsize=(12, 6))
ax = plt.subplot(111)

x_index = np.array([1, 4.5, 8, 11.5, 15, 18.5])
colors = ['#E7EAE9', '#C094B5', '#B5E0F6']
pcolors = ['#C2C2C2', '#956281', '#33AEDB']
ecolors = ['#8F8F8F', '#7A5568', '#3288AC']
labels = ['AAV9', 'AAV-ALICE.N2', 'AAV-ALICE.N6']

for i, key in enumerate(x):
    ax.bar(x_index, data[key], width=1, color=colors[i], label=labels[i])
    plt.errorbar(x_index, data[key], xerr=None, yerr=stderr[key], fmt='none', ecolor=ecolors[i], elinewidth=1,
                 capsize=2, ms=0, zorder=20, capthick=1)
    x_index += 1

# scatter_
bias = 0.2
x_index -= 3
for i, key in enumerate(x):
    for j, val in enumerate(x_index):
        position = np.linspace(val-0.5+i+bias, val+0.5+i-bias, len(scatters[key][j]))
        ax.scatter(position, scatters[key][j], c=pcolors[i], zorder=10,
                   edgecolor=ecolors[i], alpha=0.7, s=40)

# anova
above = 2
for i, region in enumerate(ref):
    start = 3.5 * i + 2
    group1 = scatters[x[0]][i]
    for j, key in enumerate(x[1:]):
        group2 = scatters[key][i]
        _, p_val = ttest_ind(group1, group2, nan_policy='omit')
        print(p_val)
        if p_val < 0.001:
            ax.text(start, np.average(group2) + above, '***', ha='center', va='bottom', fontsize=12)
        elif p_val < 0.01:
            ax.text(start, np.average(group2) + above, '**', ha='center', va='bottom', fontsize=12)
        elif p_val < 0.05:
            ax.text(start, np.average(group2) + above, '*', ha='center', va='bottom', fontsize=12)
        start += 1

plt.xticks(x_index+1, ref)
plt.ylabel('RFP/DAPI (%)')
plt.xlabel('Brain_region')
plt.title('RFP+ DAPI cells in brain cells-{}'.format(mice))
plt.legend()
plt.ylim(0, 50)

plt.show()

