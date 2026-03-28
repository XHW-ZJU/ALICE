import matplotlib.pyplot as plt
import numpy as np
from DataLoader import read_excel_data

protein_list = ['Slco1c1', 'Car4', 'ly6c1', 'ly6A', '293T']
data = read_excel_data(file_path='data0104.xlsx',
                       protein_list=protein_list,
                       virus_list=['AAV9', 'PHP.eB', 'ALICE_N2', 'ALICE_N6'])
ave_area = data[0]
print(ave_area)
bri_ness = data[1]
norm = plt.Normalize(vmin=0, vmax=12)
cmap = plt.get_cmap('RdYlBu_r')
colors = cmap(norm(bri_ness))

fig, ax = plt.subplots(figsize=(8, 6))
y = np.array([0.5]*4)
x = np.array([0.5, 1.5, 2.5, 3.5])

sc = None
min_ = []
max_ = []
for i, size in enumerate(ave_area):
    bright = bri_ness[i]
    print(bright)
    color = cmap(norm(bright))
    min_.append(min(bright))
    max_.append(max(bright))
    sc = ax.scatter(x,
                    y,
                    s=size*80,
                    c=color,
                    linewidth=1,
                    edgecolor='grey',
                    cmap=cmap,
                    zorder=10)
    y += 1

ax.set_ylim(0, 5.2)
ax.set_xlim(0, 4)
ax.grid(True, zorder=0, c='grey', alpha=0.4, axis='both')
plt.xticks(x, ['AAV9', 'PHP.eB', 'ALICE_N2', 'ALICE_N6'])
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5], protein_list)
norm = plt.Normalize(vmin=0, vmax=12)
sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
cbar = plt.colorbar(sm, ax=ax)
plt.tight_layout()
plt.show()
