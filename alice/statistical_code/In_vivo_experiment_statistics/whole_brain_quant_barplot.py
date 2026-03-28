import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

path = 'whole_brain_quant_barplot.xlsx'
df = pd.read_excel('path')
df_no_nan = df[df.Seq != np.nan]
df_no_nan['target'] = df_no_nan['RFP'] / df['DAPI'] * 100
df_no_nan = df_no_nan[df_no_nan.target != np.nan]
groupby_data = df_no_nan.groupby(['Species', 'Seq'])['target']
bar_data = groupby_data.mean()
std_err = groupby_data.sem()
scatter = groupby_data.apply(list)

x_index = np.array([1, 5.5, 10])

fig = plt.figure(figsize=(9, 4))
ax = plt.subplot(111)
mice = ['balb', 'C57', 'FVB']
virus = ['AAV9', 'ALICE_N2', 'ALICE_N6']
colors = ['#E7EAE9', '#C094B5', '#B5E0F6']
pcolors = ['#C2C2C2', '#956281', '#33AEDB']
ecolors = ['#8F8F8F', '#7A5568', '#3288AC']
labels = ['AAV9', 'AAV-ALICE.N2', 'AAV-ALICE.N6']
for i, v in enumerate(virus):
    y = [bar_data.loc[(m, v)] for m in mice]
    err = [std_err.loc[(m, v)] for m in mice]
    print(err)
    ax.bar(x_index, y, width=0.8, color=colors[i], label=labels[i])
    ax.errorbar(x_index, y, xerr=None, yerr=err, fmt='none', ecolor=ecolors[i], elinewidth=1,
                capsize=2, ms=0, zorder=20, capthick=1)
    x_index += 1

# scatter
x_index -= 3
bias = 0.2
x = ['BALB-AAV9', 'BALB-ALICE_N2', 'BALB-ALICE_N6',
     'C57-AAV9', 'C57-ALICE_N2', 'C57-ALICE_N6',
     'FVB-AAV9', 'FVB-ALICE_N2', 'FVB-ALICE_N6']
for i, key in enumerate(x_index):
    scatter_data = [scatter.loc[(m, virus[i])] for m in mice]
    for j, val in enumerate(x_index):
        position = np.linspace(val-0.5+i+bias, val+0.5+i-bias, len(scatter_data[j]))
        ax.scatter(position, scatter_data[j], c=pcolors[i], zorder=10,
                   edgecolor=ecolors[i], alpha=0.7, s=40)

# anova
above = 2
for i, m in enumerate(mice):
    start = 4.5 * i + 2
    group1 = scatter.loc[(m, virus[0])]
    for j, v in enumerate(virus[1:]):
        group2 = scatter.loc[(m, v)]
        improve = int(np.average(group2)/np.average(group1))
        ax.text(start, np.average(group2) + above + 4, str(improve), ha='center', va='bottom', fontsize=12)
        _, p_val = ttest_ind(group1, group2, nan_policy='omit')
        if p_val < 0.001:
            ax.text(start, np.average(group2) + above, '***', ha='center', va='bottom', fontsize=12)
        elif p_val < 0.01:
            ax.text(start, np.average(group2) + above, '**', ha='center', va='bottom', fontsize=12)
        elif p_val < 0.05:
            ax.text(start, np.average(group2) + above, '*', ha='center', va='bottom', fontsize=12)
        start += 1

plt.xticks(x_index+1, mice)
plt.ylabel('RFP+ brain cells(%)')
plt.title('Whole brain statistics')
plt.ylim(0, 70)
plt.legend()
plt.show()
