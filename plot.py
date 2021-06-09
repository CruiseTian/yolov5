import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab

params = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.weight': 'bold',
    # 'legend.fontsize': 40,
    # 'xtick.labelsize': 'x-large',
    # 'ytick.labelsize': 'x-large'
}
pylab.rcParams.update(params)

def plot_results():
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    aqu = np.loadtxt("aquarium.txt",usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    mush = np.loadtxt("mushroom.txt",usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    chess = np.loadtxt("chess.txt",usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    package = np.loadtxt("packages.txt",usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    vehicle = np.loadtxt("vehicles.txt",usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    cone = np.loadtxt("cone.txt",usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
    n = aqu.shape[1]  # number of rows
    x = range(0,n)
    for i in range(10):
        y1 = aqu[i, x]
        y2 = mush[i, x]
        y3 = chess[i, x]
        y4 = package[i, x]
        y5 = vehicle[i, x]
        y6 = cone[i, x]
        if i in [0, 1, 2, 5, 6, 7]:
            y1[y1 == 0] = np.nan  # don't show zero loss values
            y2[y2 == 0] = np.nan  # don't show zero loss values
            y3[y3 == 0] = np.nan  # don't show zero loss values
            y4[y4 == 0] = np.nan  # don't show zero loss values
            y5[y5 == 0] = np.nan  # don't show zero loss values
            y6[y6 == 0] = np.nan  # don't show zero loss values
            # y /= y[0]  # normalize
        ax[i].plot(x, y1, '-', c='#e41b1b', label="aquarium", linewidth=1, markersize=6)
        ax[i].plot(x, y2, '-', c='#4daf4a', label="mushroom", linewidth=1, markersize=6)
        ax[i].plot(x, y3, '-', c='#377eb8', label="chess", linewidth=1, markersize=6)
        ax[i].plot(x, y4, '-', c='#ff8000', label="packages", linewidth=1, markersize=6)
        ax[i].plot(x, y5, '-', c='#b57264', label="vehicles", linewidth=1, markersize=6)
        ax[i].plot(x, y6, '-', c='#940034', label="traffic cones", linewidth=1, markersize=6)
        ax[i].set_title(s[i],fontweight='bold')
        ax[i].legend(loc='best',fancybox=True, framealpha=0)
        ax[i].grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度
        # ax.yaxis.grid(True, linestyle='dotted')  # y坐标轴的网格使用次刻度
    plt.savefig('./results.eps')
    plt.show()

plot_results()
