# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 17:20:33 2024

@author: Muqing
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 16:59:10 2024

@author: Muqing
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Rectangle,Patch
from matplotlib.ticker import FormatStrFormatter
plt.close('all')


color_lp = 'indianred'
color_fr = 'tab:purple'
with open('STW70_FR_Ctrl_run_1200_Obj20240402.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_STW_3D_90layers = np.load(f, allow_pickle=True) # Coord_STW_3D_90layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    FR_lp_array = np.load(f, allow_pickle=True)
    FR_fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    OPCUA_Read_Time = np.load(f, allow_pickle=True) # OPCUA_Read_Time is the time duration for OPC UA communication read R varaibles from the machine within each frame collected by the IR camera during the experiment
    OPCUA_Write_Time = np.load(f, allow_pickle=True) # OPCUA_Write_Time is the time duration for OPC UA communication write R varibles to the machine within each frame collected by the IR camera during the experiment
    While_loop_time = np.load(f, allow_pickle=True) # While_loop_time is the time duration for the while loop within each frame collected by the IR camera during the experiment and it includes the frame saving time

    DPSS_angle = np.load(f, allow_pickle=True) # DPSS_angle is the adjustment of the angle on the DPSS for each layer of the step thin wall
    
    
    FR_EF = FR_lp_array*1000/FR_fr_array/3
    layer = np.arange(1, MaxT_OriginRectSearch_alllayers.shape[0]+1, dtype=int)
    


with open('STW70_LP_Ctrl_run_1200_Obj20240402.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_STW_3D_90layers = np.load(f, allow_pickle=True) # Coord_STW_3D_90layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    LP_lp_array = np.load(f, allow_pickle=True)
    LP_fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    OPCUA_Read_Time = np.load(f, allow_pickle=True) # OPCUA_Read_Time is the time duration for OPC UA communication read R varaibles from the machine within each frame collected by the IR camera during the experiment
    OPCUA_Write_Time = np.load(f, allow_pickle=True) # OPCUA_Write_Time is the time duration for OPC UA communication write R varibles to the machine within each frame collected by the IR camera during the experiment
    While_loop_time = np.load(f, allow_pickle=True) # While_loop_time is the time duration for the while loop within each frame collected by the IR camera during the experiment and it includes the frame saving time

    DPSS_angle = np.load(f, allow_pickle=True) # DPSS_angle is the adjustment of the angle on the DPSS for each layer of the step thin wall
    
    print(LP_lp_array*1000)
    print(LP_fr_array)
    LP_EF = LP_lp_array*1000/LP_fr_array/3


with open('STW70_Combined_II_1200_20240407.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_STW_3D_90layers = np.load(f, allow_pickle=True) # Coord_STW_3D_90layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    HC_lp_array = np.load(f, allow_pickle=True)
    HC_fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    OPCUA_Read_Time = np.load(f, allow_pickle=True) # OPCUA_Read_Time is the time duration for OPC UA communication read R varaibles from the machine within each frame collected by the IR camera during the experiment
    OPCUA_Write_Time = np.load(f, allow_pickle=True) # OPCUA_Write_Time is the time duration for OPC UA communication write R varibles to the machine within each frame collected by the IR camera during the experiment
    While_loop_time = np.load(f, allow_pickle=True) # While_loop_time is the time duration for the while loop within each frame collected by the IR camera during the experiment and it includes the frame saving time

    DPSS_angle = np.load(f, allow_pickle=True) # DPSS_angle is the adjustment of the angle on the DPSS for each layer of the step thin wall
    
    
    HC_EF = HC_lp_array*1000/HC_fr_array/3
    

NC_EF = [1700/1000*60/3]*91

plt.figure(figsize=(10, 8))
lns1 = plt.plot(layer, NC_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color='black',label='No Ctrl')
lns2 = plt.plot(layer, LP_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color=color_lp,label='FR Ctrl')
lns3 = plt.plot(layer, FR_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color=color_fr,label='LP Ctrl')
lns4 = plt.plot(layer, HC_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color='tab:blue',label='Hybrid Ctrl')

plt.axvspan(0, 30, facecolor='darkgrey', alpha=0.1)
plt.axvspan(30, 60, facecolor='darkgrey', alpha=0.3)
plt.axvspan(60, 90, facecolor='darkgrey', alpha=0.5)

# Set up the labels for the x and y axes
plt.xlabel('Layer #', fontweight="bold",size=17)
plt.ylabel('Energy Fluence ($J/mm^2$)', fontweight="bold",size=17)

lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
plt.legend(lns, labs,scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 2.4)

plt.yticks(np.arange(0, 45, 5))
# plt.title("STW EF of Different Control Method", fontname="Times New Roman", size=17, fontweight="bold")
plt.show()
