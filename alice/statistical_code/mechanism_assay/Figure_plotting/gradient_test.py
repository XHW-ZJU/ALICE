import matplotlib.pyplot as plt
from DataLoader import groupby_average
import numpy as np

ratio, area = groupby_average(groups=['protein', 'virus', 'group'],
                              filepath=r'.\data_mechaniam-quant.xlsx',
                              target=['AVERAGE_SIG_AREA', 'AVERAGE_TO_SIG_AREA'])

area_data = [area[area.protein == 'NONE'], area[area.protein == 'LY6A']]
bright_data = [ratio[ratio.protein == 'NONE'], ratio[ratio.protein == 'LY6A']]

virus_list = ['AAV9', 'PHP.eB']
labels = ['NONE', 'LY6A']
sc = None
ax = None
norm = plt.Normalize(vmin=0, vmax=0.35)
cmap = plt.get_cmap('RdYlBu_r')

for i in range(2):
    min_ = []
    max_ = []
    if i == 0:
        plt.figure(figsize=(4, 6))
    else:
        plt.figure(figsize=(4, 4))
    ax = plt.subplot(111)
    x = 1
    for virus in virus_list:
        size = area_data[i][area_data[i].virus == virus]['AVERAGE_TO_SIG_AREA']
        bright = bright_data[i][bright_data[i].virus == virus]['AVERAGE_SIG_AREA']
        size = size[::-1]
        min_.append(size.min())
        max_.append(size.max())
        bright = bright[::-1]
        print(size)
        colors = cmap(norm(size))
        y = np.array(range(len(size)))+1
        sc = ax.scatter(np.array([x]*len(size)),
                        y,
                        s=bright*1000,
                        c=colors,
                        linewidth=1,
                        edgecolor='grey',
                        cmap='viridis',
                        zorder=10)
        x += 1
        ax.set_xticks([1, 2], virus_list)
        if i == 0:
            ax.set_yticks(range(1, 7), ['5×10e7 vg', '1×10e8 vg', '5×10e8 vg', '1×10e9 vg', '5×10e9 vg', '1×10e10 vg'])
        else:
            ax.set_yticks(range(1, 5), ['1×10e6 vg', '5×10e6 vg', '1×10e7 vg', '5×10e7 vg'])
        ax.set_xlim(0, 3)
        if i == 0:
            ax.set_ylim(0, 7)
        else:
            ax.set_ylim(0, 5)
        ax.grid(True, zorder=0, c='grey', alpha=0.4, axis='both')
    norm1 = plt.Normalize(vmin=0, vmax=0.35)
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm1)
    cbar = plt.colorbar(sm, ax=ax)
    plt.tight_layout()
    plt.show()


