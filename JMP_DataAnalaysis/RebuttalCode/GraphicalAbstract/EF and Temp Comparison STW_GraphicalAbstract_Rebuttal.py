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
import scipy.stats as st
plt.close('all')


color_lp = 'indianred'
color_fr = 'tab:purple'

NC_Temp = np.load('SPTW_NOCtrl_01312024_Medium_temperature_RecSearch_5(row)x31(column).npy', allow_pickle=True)

def stat_cal(temp,title):
    mid_temp_mean = np.mean(temp[30:60])
    mid_temp_sd = np.std(temp[30:60])
    short_temp_mean = np.mean(temp[60:90])
    short_temp_sd = np.std(temp[60:91])
    mid_data  = temp[30:60]
    short_data = temp[60:91]
    print('************************')
    mid_lower, mid_upper = st.t.interval(confidence=0.95,  df=len(mid_data)-1,
                 loc=np.mean(mid_data), 
                 scale=st.sem(mid_data))
    mid_CI =  (mid_upper-mid_lower)/2
    print(mid_lower, '+', mid_upper)

 
    
    short_lower, short_upper = st.t.interval(confidence=0.95, df=len(short_data)-1,
                 loc=np.mean(short_data), 
                 scale=st.sem(short_data))
    short_CI =  (short_upper-short_lower)/2
    print(short_lower, '+', short_upper)
    print(title+' middle mean:'+ str(mid_temp_mean))
    print(title+' middle std:'+str(mid_temp_sd))
    print(title+' short mean:'+str(short_temp_mean))
    print(title+' short std:'+str(short_temp_sd))
    # return mid_temp_mean,mid_temp_sd,short_temp_mean,short_temp_sd, mid_lower, mid_upper, short_lower, short_upper
    return mid_temp_mean,mid_temp_sd,short_temp_mean,short_temp_sd, mid_CI, short_CI

with open('STW70_FR_Ctrl_run_1200_Obj20240402.npy', 'rb') as f:
    FR_Temp = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
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
    
    FR_Temp = np.median(FR_Temp, axis=1) 
    FR_EF = FR_lp_array*1000/FR_fr_array/3
    layer = np.arange(1, FR_Temp.shape[0]+1, dtype=int)
    


with open('STW70_LP_Ctrl_run_1200_Obj20240402.npy', 'rb') as f:
    LP_Temp = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
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
    
    LP_Temp = np.median(LP_Temp, axis=1) 
    LP_EF = LP_lp_array*1000/LP_fr_array/3


with open('STW70_Combined_II_1200_20240407.npy', 'rb') as f:
    HC_Temp = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
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
    
    HC_Temp = np.median(HC_Temp, axis=1) 
    HC_EF = HC_lp_array*1000/HC_fr_array/3
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.family"] = "Times New Roman" 

   
# %% Stat Cal
# NC_mid_temp_mean,NC_mid_temp_sd, NC_short_temp_mean,NC_short_temp_sd, NC_mid_CI_lower,NC_mid_CI_upper, NC_short_CI_lower,NC_short_CI_upper = stat_cal(NC_Temp,'NC')
# LP_mid_temp_mean,LP_mid_temp_sd, LP_short_temp_mean,LP_short_temp_sd, LP_mid_CI_lower,LP_mid_CI_upper, LP_short_CI_lower,LP_short_CI_upper = stat_cal(LP_Temp,'LP')
# FR_mid_temp_mean,FR_mid_temp_sd, FR_short_temp_mean,FR_short_temp_sd, FR_mid_CI_lower,FR_mid_CI_upper, FR_short_CI_lower,FR_short_CI_upper = stat_cal(FR_Temp,'FR')
# HC_mid_temp_mean,HC_mid_temp_sd, HC_short_temp_mean,HC_short_temp_sd, HC_mid_CI_lower,HC_mid_CI_upper, HC_short_CI_lower,HC_short_CI_upper = stat_cal(HC_Temp,'HC')

NC_mid_temp_mean,NC_mid_temp_sd, NC_short_temp_mean,NC_short_temp_sd, NC_mid_CI, NC_short_CI = stat_cal(NC_Temp,'NC')
LP_mid_temp_mean,LP_mid_temp_sd, LP_short_temp_mean,LP_short_temp_sd, LP_mid_CI, LP_short_CI = stat_cal(LP_Temp,'LP')
FR_mid_temp_mean,FR_mid_temp_sd, FR_short_temp_mean,FR_short_temp_sd, FR_mid_CI, FR_short_CI = stat_cal(FR_Temp,'FR')
HC_mid_temp_mean,HC_mid_temp_sd, HC_short_temp_mean,HC_short_temp_sd, HC_mid_CI, HC_short_CI = stat_cal(HC_Temp,'HC')


Ctrl_mid_temp_mean= [NC_mid_temp_mean, LP_mid_temp_mean, FR_mid_temp_mean, HC_mid_temp_mean]
Ctrl_mid_temp_sd =[NC_mid_temp_sd, LP_mid_temp_sd, FR_mid_temp_sd, HC_mid_temp_sd]
Ctrl_short_temp_mean= [NC_short_temp_mean, LP_short_temp_mean, FR_short_temp_mean, HC_short_temp_mean]
Ctrl_short_temp_sd =[NC_short_temp_sd, LP_short_temp_sd, FR_short_temp_sd, HC_short_temp_sd]

fig, ax1 = plt.subplots(figsize=(10, 8)) 
Method = ["No Ctrl","LP Ctrl", "FR Ctrl", "Hybrid Ctrl"]

bar_width = 0.4
x = np.arange(len(Method))
# Create the figure and axes

# ax2= ax1.twinx()


bar1 = ax1.bar(x - bar_width/2, Ctrl_mid_temp_mean,yerr=Ctrl_mid_temp_sd, 
               color ="orange", width = 0.4, label='Medium Heat Cycle',error_kw=dict(lw=5, capsize=10, capthick=2))
bar2 = ax1.bar(x + bar_width/2, Ctrl_short_temp_mean,yerr=Ctrl_short_temp_sd,
               color ="grey",width = 0.4, label ='Short Heat Cycle',error_kw=dict(lw=5, capsize=10, capthick=2))
lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
plt.ylim(0,1700)
ax1.set_xticks(x)
ax1.set_xticklabels(Method,fontsize=30)
ax1.set_ylabel('Temperature (°C)', fontsize=30, fontweight='bold')
# ax2.set_ylabel("Width Value(mm)")

legend=ax1.legend(lines, labels, loc='best', ncol=1,fontsize=23.5,framealpha=0.3)
ax1.tick_params(axis='both', labelcolor='black',labelsize=30)

# Make the legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Make tick labels bold for ax1
for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
plt.tight_layout()
# plt.show()



# %% Stat Cal 2
NC_mid_temp_mean,NC_mid_temp_sd, NC_short_temp_mean,NC_short_temp_sd, NC_mid_CI, NC_short_CI = stat_cal(NC_Temp,'NC')
LP_mid_temp_mean,LP_mid_temp_sd, LP_short_temp_mean,LP_short_temp_sd, LP_mid_CI, LP_short_CI = stat_cal(LP_Temp,'LP')
FR_mid_temp_mean,FR_mid_temp_sd, FR_short_temp_mean,FR_short_temp_sd, FR_mid_CI, FR_short_CI = stat_cal(FR_Temp,'FR')
HC_mid_temp_mean,HC_mid_temp_sd, HC_short_temp_mean,HC_short_temp_sd, HC_mid_CI, HC_short_CI = stat_cal(HC_Temp,'HC')


Ctrl_mid_temp_mean= [NC_mid_temp_mean, LP_mid_temp_mean, FR_mid_temp_mean, HC_mid_temp_mean]
Ctrl_mid_temp_sd =[NC_mid_temp_sd, LP_mid_temp_sd, FR_mid_temp_sd, HC_mid_temp_sd]
Ctrl_short_temp_mean= [NC_short_temp_mean, LP_short_temp_mean, FR_short_temp_mean, HC_short_temp_mean]
Ctrl_short_temp_sd =[NC_short_temp_sd, LP_short_temp_sd, FR_short_temp_sd, HC_short_temp_sd]
Temp_mean_data = [[NC_mid_temp_mean,NC_short_temp_mean],[LP_mid_temp_mean,LP_short_temp_mean],
                  [FR_mid_temp_mean,FR_short_temp_mean],[HC_mid_temp_mean,HC_short_temp_mean]]
Temp_sd_data = [[NC_mid_temp_sd,NC_short_temp_sd],[LP_mid_temp_sd,LP_short_temp_sd],
                  [FR_mid_temp_sd,FR_short_temp_sd],[HC_mid_temp_sd,HC_short_temp_sd]]
Temp_CI_data = [[NC_mid_CI,NC_short_CI],[LP_mid_CI,LP_short_CI],
                  [FR_mid_CI,FR_short_CI],[HC_mid_CI,HC_short_CI]]

print(f'Confident Interval for No Ctrl: {NC_mid_CI}, {NC_short_CI}')
print(f'Confident Interval for LP Ctrl: {LP_mid_CI}, {LP_short_CI}')
print(f'Confident Interval for FR Ctrl: {FR_mid_CI}, {FR_short_CI}')
print(f'Confident Interval for Hybrid Ctrl: {HC_mid_CI}, {HC_short_CI}')


fig, ax1 = plt.subplots(figsize=(10, 8)) 
Method = ["No Ctrl","LP Ctrl", "FR Ctrl", "Hybrid Ctrl"]
Category=["Medium Heat Cycle", "Short Heat Cycle"]
Color = ['black',color_lp, color_fr, 'tab:blue']
bar_width = 0.2
x = np.arange(len(Category))
# Create the figure and axes

# ax2= ax1.twinx()
for i in range(len(Method)):
    ax1.bar(x + i * bar_width, Temp_mean_data[i],bar_width, yerr=Temp_sd_data[i], label=Method[i],
            color =Color[i], error_kw=dict(lw=5, capsize=10, capthick=2, ecolor="red"),alpha=0.8)
legend = ax1.legend()

# ax1.set_ylabel('Values')
# ax1.set_title('Grouped Bar Chart with Error Bars')
# ax1.set_xticks(x + 1.5 * bar_width)
# ax1.set_xticklabels(Method)

# bar1 = ax1.bar(x - bar_width/2, Ctrl_mid_temp_mean,yerr=Ctrl_mid_temp_sd, 
#                color ="orange", width = 0.4, label='Middle Heat Cycle',error_kw=dict(lw=5, capsize=10, capthick=2))
# bar2 = ax1.bar(x + bar_width/2, Ctrl_short_temp_mean,yerr=Ctrl_short_temp_sd,
#                color ="grey",width = 0.4, label ='Short Heat Cycle',error_kw=dict(lw=5, capsize=10, capthick=2))
lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
plt.ylim(1000,1550)
ax1.set_xticks(x + 1.5 * bar_width)
ax1.set_xticklabels(Category,fontsize=30)
ax1.set_ylabel('Temperature (°C)', fontsize=30, fontweight='bold')
# ax2.set_ylabel("Width Value(mm)")

legend=ax1.legend(lines, labels, loc='upper left', ncol=2,fontsize=23.5,framealpha=0.3)
ax1.tick_params(axis='both', labelcolor='black',labelsize=30)

for text, color in zip(legend.get_texts(), Color):
     text.set_fontweight('bold')
     text.set_color(color)
# Make tick labels bold for ax1
for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
plt.tight_layout()
plt.show()

# Confident Interval Plot
fig, ax1 = plt.subplots(figsize=(10, 8)) 
Method = ["No Ctrl","LP Ctrl", "FR Ctrl", "Hybrid Ctrl"]
Category=["Medium Heat Cycle", "Short Heat Cycle"]
Color = ['black',color_lp, color_fr, 'tab:blue']
bar_width = 0.2
x = np.arange(len(Category))
# Create the figure and axes

# ax2= ax1.twinx()
for i in range(len(Method)):
    ax1.bar(x + i * bar_width, Temp_mean_data[i],bar_width, yerr=Temp_CI_data[i], label=Method[i],
            color =Color[i], error_kw=dict(lw=5, capsize=10, capthick=2, ecolor="red"),alpha=0.8)
legend = ax1.legend()

# ax1.set_ylabel('Values')
# ax1.set_title('Grouped Bar Chart with Error Bars')
# ax1.set_xticks(x + 1.5 * bar_width)
# ax1.set_xticklabels(Method)

# bar1 = ax1.bar(x - bar_width/2, Ctrl_mid_temp_mean,yerr=Ctrl_mid_temp_sd, 
#                color ="orange", width = 0.4, label='Middle Heat Cycle',error_kw=dict(lw=5, capsize=10, capthick=2))
# bar2 = ax1.bar(x + bar_width/2, Ctrl_short_temp_mean,yerr=Ctrl_short_temp_sd,
#                color ="grey",width = 0.4, label ='Short Heat Cycle',error_kw=dict(lw=5, capsize=10, capthick=2))
lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
plt.ylim(1000,1550)
ax1.set_xticks(x + 1.5 * bar_width)
ax1.set_xticklabels(Category,fontsize=30)
ax1.set_ylabel('Temperature (°C)', fontsize=30, fontweight='bold')
# ax2.set_ylabel("Width Value(mm)")

legend=ax1.legend(lines, labels, loc='upper left', ncol=2,fontsize=23.5,framealpha=0.3)
ax1.tick_params(axis='both', labelcolor='black',labelsize=30)

for text, color in zip(legend.get_texts(), Color):
     text.set_fontweight('bold')
     text.set_color(color)
# Make tick labels bold for ax1
for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
plt.rc('font', weight='bold')
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
plt.tight_layout()
plt.show()





# %% EF
NC_EF = [1700/1000*60/3]*91


fig, ax1 = plt.subplots(figsize=(10, 8))  # Create a new figure with a custom size

ax1.axvspan(0, 30, facecolor='darkgrey', alpha=0.1)
ax1.axvspan(30, 60, facecolor='darkgrey', alpha=0.3)
ax1.axvspan(60, 90, facecolor='darkgrey', alpha=0.5)


lns1 = plt.plot(layer, NC_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color='black',label='No Ctrl EF')
lns2 = plt.plot(layer, LP_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color=color_lp,label='FR Ctrl EF')
lns3 = plt.plot(layer, FR_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color=color_fr,label='LP Ctrl EF')
lns4 = plt.plot(layer, HC_EF[1:], linewidth=3, marker='.', markersize=10, linestyle='-', color='tab:blue',label='Hybrid EF')
plt.xlabel('Layer #',fontweight='bold',size=30)
plt.ylabel('Energy Fluence ($J/mm^2$)',fontweight='bold',size=30)
ax1.tick_params(axis='both', labelcolor='black',labelsize=30)
plt.yticks(np.arange(0, 40 + 5, 5))

# Add a legend
lns = lns1+lns2+lns3+lns4
labs = [l.get_label() for l in lns]
legend = plt.legend(lns, labs, loc='lower center',fontsize=23.5,ncol=2, bbox_to_anchor=(0.5, 0.0), framealpha=0.3)

# Set up the title for the plot
# plt.title('Medium temperature acqureied by different search methods')
# plt.title("FR Control Step Thin Wall with 1200$^\circ$C Objective", fontname="Times New Roman",
#           size=30, fontweight="bold")
# Make the legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set up the title for the plot
# plt.title('Medium temperature acqureied by different search methods')
# plt.title("LP Control Thin Wall with 1200$^\circ$C Objective", fontname="Times New Roman",
          # size=30, fontweight="bold")

# Make tick labels bold for ax1
for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')


# Add grid lines

plt.rcParams["axes.labelweight"] = "bold"

# plt.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.99)
#********************** Plot medium temperature - END
plt.tight_layout() 
plt.grid(axis='y')
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.15)
# plt.title("TW EF of Different Control Method", fontname="Times New Roman",
#           size=17, fontweight="bold")
plt.show()

# %% Temp
color_T = 'tab:blue'
color_objT = 'blue'

fig, ax1 = plt.subplots(figsize=(10, 8))  # Create a new figure with a custom size

ax1.axvspan(0, 30, facecolor='darkgrey', alpha=0.1)
ax1.axvspan(30, 60, facecolor='darkgrey', alpha=0.3)
ax1.axvspan(60, 90, facecolor='darkgrey', alpha=0.5)


lns1 = plt.plot(layer, NC_Temp, linewidth=3, marker='.', markersize=10, linestyle='-', color='black',label='No Ctrl')
lns2 = plt.plot(layer, LP_Temp, linewidth=3, marker='.', markersize=10, linestyle='-', color=color_lp,label='FR Ctrl')
lns3 = plt.plot(layer, FR_Temp, linewidth=3, marker='.', markersize=10, linestyle='-', color=color_fr,label='LP Ctrl')
lns4 = plt.plot(layer, HC_Temp, linewidth=3, marker='.', markersize=10, linestyle='-', color='tab:blue',label='Hybrid Ctrl')
lns5 = plt.plot(layer, np.full((90, ), 1200),linestyle='dashed', linewidth=3, color=color_objT, label='$T_{obj}$')
plt.xlabel('Layer #',fontweight='bold',size=30)
plt.ylabel('Median Temperature (°C)',fontweight='bold',size=30)
ax1.tick_params(axis='both', labelcolor='black',labelsize=30)

# Add a legend
lns = lns1+lns2+lns3+lns4+lns5
labs = [l.get_label() for l in lns]
legend = plt.legend(lns, labs, loc='lower right',fontsize=23.5,ncol=2, framealpha=0.3)

# Set up the title for the plot
# plt.title('Medium temperature acqureied by different search methods')
# plt.title("FR Control Step Thin Wall with 1200$^\circ$C Objective", fontname="Times New Roman",
#           size=30, fontweight="bold")
# Make the legend text bold
for text in legend.get_texts():
    text.set_fontweight('bold')

# Set up the title for the plot
# plt.title('Medium temperature acqureied by different search methods')
# plt.title("LP Control Thin Wall with 1200$^\circ$C Objective", fontname="Times New Roman",
          # size=30, fontweight="bold")

# Make tick labels bold for ax1
for tick in ax1.xaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')
for tick in ax1.yaxis.get_major_ticks():
    tick.label1.set_fontweight('bold')


# Add grid lines

plt.rcParams["axes.labelweight"] = "bold"

# plt.subplots_adjust(left=0.12, right=0.88, bottom=0.1, top=0.99)
#********************** Plot medium temperature - END
plt.tight_layout() 
plt.grid(axis='y')
plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.15)
# plt.title("TW EF of Different Control Method", fontname="Times New Roman",
#           size=17, fontweight="bold")
plt.show()
