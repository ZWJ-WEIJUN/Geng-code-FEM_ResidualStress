# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:24:58 2024

@author: Muqing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
plt.rcParams["font.family"] = "Times New Roman"



fig, ax = plt.subplots(figsize=(10, 8))
#In order of No-control, LP CTRL, FR CTRL, Hybrid
Width_target = 3.15
Height_target = 63

Width= [3.497, 3.312, 3.307, 3.222]
Width_SD =[0.184, 0.160, 0.078,0.092]
Height = [62.1,63.4,64.22,64.03]
Height_SD = [0.752,0.449,0.336,0.330]
Time = [10.463,10.463,9.842,9.703]
graph_color = ['black', 'indianred','tab:purple', 'tab:blue']
tag = ['No CTRl', 'LP CTRL', 'FR CTRL', 'Hybrid CTRL']


for i in range(len(Width)):
    plt.errorbar(Width[i], Height[i], xerr=Width_SD[i], yerr=Height_SD[i], fmt="o", ecolor=graph_color[i],markerfacecolor=graph_color[i],
                 markeredgecolor=graph_color[i],capsize=5, label= tag[i])
    plt.xlim(3.0, 3.8)
    plt.ylim(60, 65)
    plt.xticks(np.arange(3.0, 3.8, 0.1))
    plt.yticks(np.arange(60, 65.5,0.5))


plt.scatter(Width_target, Height_target, color='orange', label='target temperture')
plt.legend()
plt.show()