# plot_cdfs.py
# Author: Noah Athens
# Created: April 13th, 2018

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def subplots_dim(n):
    nrows = 1
    ncols = 1
    while True:
        if n > (nrows * ncols):
            ncols += 1
        else:
            break
        if n > (nrows * ncols):
            nrows += 1
        else:
            break
    return nrows, ncols

def plot_CDFs(data, labels, parameters = 'all'):
    if parameters == 'all': parameters = data.columns.values
    num = len(parameters)
    nrows, ncols = subplots_dim(num)
    fig = plt.figure()
    for i, name in enumerate(parameters):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        x = np.sort(data[name].values)
        y = np.linspace(0, 1, x.shape[0])
        ax.plot(x, y, 'k-')
        minval = x.min()
        maxval = x.max()
        ax.set_title(name, fontsize=10)
        for cluster in np.sort(labels.unique()):
            x = np.sort(data.loc[labels == cluster, name].values)
            x = np.insert(x, 0, minval)
            x = np.append(x, maxval)
            y = np.linspace(0, 1, x.shape[0])
            ax.plot(x, y, '-')
    fig.subplots_adjust(hspace=-1, wspace=-1)
    fig.tight_layout()
    plt.show()
    return None
