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




#In order of No-control, LP CTRL, FR CTRL, Hybrid
Width_target = 3
Height_target = 63

Width= [3.724, 3.286, 3.378, 3.168]
Width_SD =[0.148, 0.132, 0.0793,0.0993]
Width_CI = [0.0388, 0.0345, 0.0208, 0.0260]
Height = [63.091,63.977,64.457,63.697]
Height_SD = [0.613,0.468,0.475,0.438]
Height_CI  = [0.0244, 0.0183, 0.0189, 0.0171]
Time = [8.2,8.2,7.36,7.57]
graph_color = ['black', 'indianred','tab:purple', 'tab:blue']
tag = ['No CTRl', 'LP CTRL', 'FR CTRL', 'Hybrid CTRL']

y_min = 60
y_max = 66.5
x_min = 2.35
x_max = 4.1
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(Width)):
    plt.errorbar(Width[i], Height[i], xerr=Width_SD[i], yerr=Height_SD[i], fmt="o", ecolor=graph_color[i],markerfacecolor=graph_color[i],
                 markeredgecolor=graph_color[i],capsize=5, label= tag[i])
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xticks(np.arange(x_min+0.05, x_max+0.05, 0.2))
    plt.yticks(np.arange(y_min+1,y_max+0.5,1))



ax.axvspan(0, Width_target, (1-(y_max-Height_target)/(y_max-y_min)), 1, color="yellow",alpha=0.1)
ax.axvspan(Width_target,x_max, 0, (1-(y_max-Height_target)/(y_max-y_min)), color="yellow",alpha=0.1)
ax.axvspan(Width_target,x_max,(1-(y_max-Height_target)/(y_max-y_min)), 1, color="green",alpha=0.1)
ax.axvspan(0, Width_target, 0, (1-(y_max-Height_target)/(y_max-y_min)), color="red",alpha=0.1)

plt.scatter(Width_target, Height_target, color='orange', marker= '*', s= 400)

ax.annotate("No Ctrl\nTime: 8.2 mins", xy=(Width[0], Height[0]), fontsize = 23.5, xytext= (Width[0]-0.5, Height[0]-1.0),
            arrowprops= dict(color='black', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='black'), 
            color='black')
ax.annotate("LP Ctrl\nTime: 8.2 mins", xy=(Width[1], Height[1]), fontsize = 23.5, xytext= (Width[1]+0.25, Height[1]+0.35),
            arrowprops= dict(color='indianred', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='indianred'), 
            color='indianred')
ax.annotate("FR Ctrl\nTime: 7.36 mins", xy=(Width[2], Height[2]), fontsize = 23.5, xytext= (Width[2]+0.1, Height[2]+0.75),
            arrowprops= dict(color='tab:purple', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='tab:purple'), 
            color='tab:purple')
ax.annotate("Hybrid Ctrl\nTime: 7.57 mins", xy=(Width[3], Height[3]), fontsize = 23.5, xytext= (Width[3]-0.68, Height[3]+0.55),
            arrowprops= dict(color='tab:blue', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='tab:blue'), 
            color='tab:blue')
ax.annotate("Target", xy=(Width_target, Height_target), fontsize = 27, xytext= (Width_target, Height_target-0.5),
            bbox=dict(boxstyle="round", fc="w",color='orange'), 
            color='orange', ha='center') 
ax.annotate("Unacceptable\n(Underbuilt in Width)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_min+0.01, y_max-0.1),horizontalalignment='left',verticalalignment='top' )
ax.annotate("Acceptable\n(Overbuilt)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_max-0.01, y_max-0.1),horizontalalignment='right',verticalalignment='top' )
ax.annotate("Unacceptable\n(Underbuilt in Height)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_max-0.01, y_min+0.1),horizontalalignment='right',verticalalignment='bottom' )
ax.annotate("Unacceptable\n(Underbuilt in Height\nand Width)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_min+0.01, y_min+0.1),horizontalalignment='left',verticalalignment='bottom')



plt.xlabel('Width (mm)',fontweight='bold',size=30)
plt.ylabel('Height (mm)',fontweight='bold',size=30)
# plt.ylim(61.5, 64.5)
ax.tick_params(axis='both', labelcolor='black',labelsize=30)

plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
plt.tight_layout()
# plt.legend(fontsize=17, loc= 'lower left')
plt.show()



# %% Confident Interval
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(Width)):
    plt.errorbar(Width[i], Height[i], xerr=Width_CI[i], yerr=Height_CI[i], fmt="o", ecolor=graph_color[i],markerfacecolor=graph_color[i],
                 markeredgecolor=graph_color[i],capsize=5, label= tag[i])
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xticks(np.arange(x_min+0.05, x_max+0.05, 0.2))
    plt.yticks(np.arange(y_min+1,y_max+0.5,1))

ax.axvspan(0, Width_target, (1-(y_max-Height_target)/(y_max-y_min)), 1, color="yellow",alpha=0.1)
ax.axvspan(Width_target,x_max, 0, (1-(y_max-Height_target)/(y_max-y_min)), color="yellow",alpha=0.1)
ax.axvspan(Width_target,x_max,(1-(y_max-Height_target)/(y_max-y_min)), 1, color="green",alpha=0.1)
ax.axvspan(0, Width_target, 0, (1-(y_max-Height_target)/(y_max-y_min)), color="red",alpha=0.1)

plt.scatter(Width_target, Height_target, color='orange', marker= '*', s= 400)

ax.annotate("No Ctrl\nTime: 8.2 mins", xy=(Width[0], Height[0]-0.1), fontsize = 23.5, xytext= (Width[0]-0.3, Height[0]-1.0),
            arrowprops= dict(color='black', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='black'), 
            color='black')
ax.annotate("LP Ctrl\nTime: 8.2 mins", xy=(Width[1]+0.05, Height[1]), fontsize = 23.5, xytext= (Width[1]+0.3, Height[1]+0.3),
            arrowprops= dict(color='indianred', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='indianred'), 
            color='indianred')
ax.annotate("FR Ctrl\nTime: 7.36 mins", xy=(Width[2], Height[2]+0.1), fontsize = 23.5, xytext= (Width[2]+0.1, Height[2]+0.75),
            arrowprops= dict(color='tab:purple', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='tab:purple'), 
            color='tab:purple')
ax.annotate("Hybrid Ctrl\nTime: 7.57 mins", xy=(Width[3]-0.05, Height[3]), fontsize = 23.5, xytext= (Width[3]-0.68, Height[3]+0.55),
            arrowprops= dict(color='tab:blue', shrink=0.05, width=2, headwidth=10, headlength=10),
            bbox=dict(boxstyle="round", fc="w",color='tab:blue'), 
            color='tab:blue')
ax.annotate("Target", xy=(Width_target, Height_target), fontsize = 27, xytext= (Width_target, Height_target-0.5),
            bbox=dict(boxstyle="round", fc="w",color='orange'), 
            color='orange', ha='center') 
ax.annotate("Unacceptable\n(Underbuilt in Width)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_min+0.01, y_max-0.1),horizontalalignment='left',verticalalignment='top' )
ax.annotate("Acceptable\n(Overbuilt)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_max-0.01, y_max-0.1),horizontalalignment='right',verticalalignment='top' )
ax.annotate("Unacceptable\n(Underbuilt in Height)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_max-0.01, y_min+0.1),horizontalalignment='right',verticalalignment='bottom' )
ax.annotate("Unacceptable\n(Underbuilt in Height\nand Width)", xy=(x_min, y_max), fontsize = 23.5, xytext= (x_min+0.01, y_min+0.1),horizontalalignment='left',verticalalignment='bottom')



plt.xlabel('Width (mm)',fontweight='bold',size=30)
plt.ylabel('Height (mm)',fontweight='bold',size=30)
# plt.ylim(61.5, 64.5)
ax.tick_params(axis='both', labelcolor='black',labelsize=30)

plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
plt.tight_layout()
# plt.legend(fontsize=17, loc= 'lower left')
plt.show()