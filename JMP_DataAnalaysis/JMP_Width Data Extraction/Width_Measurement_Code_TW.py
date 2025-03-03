# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:32:31 2024

@author: Muqing
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 29 22:46:32 2024

@author: muqin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')
plt.rcParams["font.family"] = "Times New Roman"


def clean_data_v1(file_path, expected_number_of_fields):
    
    # Variables definition and initialization
    #For one wall
    cleaned_out_lines = []
    Invalidlines = []    
    # Open and read the CSV file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Identify lines to clean out and split the data
    for i, line in enumerate(lines):
        if len(line.split(',')) == expected_number_of_fields:
            pass
        else:
            cleaned_out_lines.append((i+1, line))
            print(f"Line {i+1} has {len(line.split(','))} fields instead of {expected_number_of_fields}")
            print(f"Line {i+1}: {line}")
    print(cleaned_out_lines)
    
    # Split the data into two sections based on invalid lines
    section1_lines = lines[cleaned_out_lines[0][0]:cleaned_out_lines[1][0]-1]
    
    print(f"Section 1 has {len(section1_lines)} lines")
    # print(f"Section 2 has {len(section2_lines)} lines")
    
    # Create new variables to store the cleaned data
    Wall = section1_lines
    # Right_Wall = section2_lines
    return Wall

def width_cal(cleanData,layer_target,y_range,y_inc):
    
    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
    df = pd.DataFrame([x.split(',') for x in cleanData]) # Convert the data into a DataFrame
    
    df = df.map(str.strip)  # Remove leading and trailing whitespace characters
    

    df = df.astype(float)  # Convert the data into float
    # print(df)
    df.iloc[:, 1] = df.iloc[:, 1].abs()
    df_original = df
    # plt.show()
    y_min = 3
    y_max = 60
    
    # y_range = 15
    # y_inc = 0.5
    
    df =df[(df[1] > y_min) & (df[1] < y_max)]
    
    left_x = df[0].min()
    right_x =df[0].max()
    mid_x = (left_x+right_x)/2
    
    df_left = df[(df[0] > left_x-1.2) & (df[0] < left_x+1.2)]
    df_right = df[(df[0] > right_x-1.2) & (df[0] < right_x+1.2)]
    if (len(df_left)+len(df_right)!=len(df)):
        quit()
    
    left_x_mean = []
    right_x_mean = []    
    
    
    Height_info = []
  
    for i in layer_target:
        for j in np.arange(i, i+y_range,y_inc):
            # cal_left = df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)][1]
            # print(len(cal_left))
            Height_info.append(j)
            cal_left = df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)][0].mean()
            left_x_mean.append(cal_left)
            # cal_right = df_right[(df_right[1] > j) & (df_right[1] < j+1)][1]
            # right_x_mean.append(cal_right)
            cal_right = df_right[(df_right[1] > j) & (df_right[1] < j+y_inc)][0].mean()
            right_x_mean.append(cal_right)
    
    left_x_mean = np.array(left_x_mean)
    right_x_mean = np.array(right_x_mean)
    Width = right_x_mean - left_x_mean 
    Width_mean = np.mean(Width)
    Width_SD = np.std(Width)        
    return Width,Height_info, Width_mean, Width_SD, df_original, mid_x

def plot_graph(df1, df2, graph):
    df1_x = df1[0]
    df1_y = df1[1]
    df2_x = df2[0]
    df2_y = df2[1]   
    fig, ax = plt.subplots(figsize=(10, 8))  
    ax.scatter(df1_x,df1_y, color=graph[0],label=graph[1])
    ax.scatter(df2_x-55,df2_y, color=graph[2],label=graph[3])

    plt.title('Step Thin Wall Side Profile',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.ylim(0,75)
    plt.xlabel('Position x (mm)', fontweight="bold",size=17)
    plt.ylabel('Position y (mm)', fontweight="bold",size=17)
    plt.legend(scatterpoints=1,loc='upper center',ncol=2,fontsize=15,markerscale = 1.0)
    


if __name__ == '__main__':
    # Call the clean_data function
    y_range= 60
    y_inc = 1
    Height_target= [3] 
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    expected_number_of_fields = 3
    

    
            
    # file_path_NC = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_NoCtrl_ThicknessProfile.csv'
    # file_path_LP = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_LPCtrl_ThicknessProfile.csv'
    # file_path_FR = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_FRCtrl_ThicknessProfile.csv'
    # file_path_HC = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_Hybrid3Ctrl_ThicknessProfile.csv'

            
    file_path_NC = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_NoCtrl_ThicknessProfile.csv'
    file_path_LP = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_LPCtrl_ThicknessProfile.csv'
    file_path_FR = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_FRCtrl_ThicknessProfile.csv'
    file_path_HC = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_Hybrid3Ctrl_ThicknessProfile.csv'


    # file_path_NC = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/TW25mm_NoCtrl_ThicknessProfile.csv'
    # file_path_LP = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/TW25mm_LPCtrl_ThicknessProfile.csv'
    # file_path_FR = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/TW25mm_FRCtrl_ThicknessProfile.csv'
    # file_path_HC = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/TW25mm_Hybrid3Ctrl_ThicknessProfile.csv'




    TW25_NC = clean_data_v1(file_path_NC, expected_number_of_fields)
    TW25_LP = clean_data_v1(file_path_LP, expected_number_of_fields)
    TW25_FR = clean_data_v1(file_path_FR, expected_number_of_fields)   
    TW25_HC = clean_data_v1(file_path_HC, expected_number_of_fields)

    

    No_Ctrl_width,Height_info,No_Ctrl_mean,No_Ctrl_SD,No_Ctrl_df, No_Ctrl_mid_x = width_cal(TW25_NC,Height_target,y_range,y_inc)
    LP_width,Height_info,LP_mean,LP_SD,LP_df,LP_mid_x = width_cal(TW25_LP,Height_target,y_range,y_inc)
    FR_width,Height_info,FR_mean,FR_SD,FR_df,FR_mid_x = width_cal(TW25_FR,Height_target,y_range,y_inc)
    HC_width,Height_info, HC_mean, HC_SD,HC_df,HC_mid_x = width_cal(TW25_HC,Height_target,y_range,y_inc)

    
    print('No Ctrl Mean:', No_Ctrl_mean)
    print('No Ctrl SD:', No_Ctrl_SD)
    print('LP 1200 Mean:', LP_mean)
    print('LP 1200 SD:', LP_SD)
    print('FR 1200 Mean:',FR_mean)
    print('FR 1200 SD:', FR_SD)

    print('Hybrid 1200 Mean:', HC_mean)
    print('Hybrid 1200 SD:', HC_SD)
    
# %% **********************Comparison    
    
    plot_graph(LP_df,FR_df,[color_lp,'LP Control 1200$^\circ$C Objective',color_fr,'FR Control 1200$^\circ$C Objective'])

#**************************Scatter Plot***************************************************     
    fig, ax = plt.subplots(figsize=(10, 8))  
    No_Ctrl = ax.scatter(Height_info,No_Ctrl_width, color='black',label='TW 25 No Control')
    Lp = ax.scatter(Height_info,LP_width, color=color_lp, label='TW 25 LP Control')
    Fr = ax.scatter(Height_info,FR_width, color=color_fr, label='TW 25 FR Control')
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW 25 Combined I Control')
    HC = ax.scatter(Height_info,HC_width, color='tab:blue', label='TW 25 HybridControl')
    
    plt.ylim(2.4,4.5)
    plt.title('Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)

    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

 #*************************Line in section Plot TW 25 Comparison***************************************************    
    # Initialize arrays to store the elements
    no_ctrl_array = []
    lp_array = []
    fr_array = []
    hc_array = []
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)): 

        no_ctrl_segment = No_Ctrl_width[i*y_range:(i+1)*y_range]
        lp_segment = LP_width[i*y_range:(i+1)*y_range]
        fr_segment = FR_width[i*y_range:(i+1)*y_range]
        hc_segment = HC_width[i*y_range:(i+1)*y_range]

        # Append segments to arrays
        no_ctrl_array.extend(no_ctrl_segment)
        lp_array.extend(lp_segment)
        fr_array.extend(fr_segment)
        hc_array.extend(hc_segment)

        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='No Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        Lp = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_width[i*y_range:(i+1)*y_range], color=color_lp, label='LP Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        Fr = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='FR Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW 25 Combined I Control')
        HC = ax.plot(Height_info[i*y_range:(i+1)*y_range],HC_width[i*y_range:(i+1)*y_range], color='tab:blue', label='Hybrid Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
    
    # plt.ylim(2.4,4.5)

    plt.xlim(0, 65)
    plt.ylim(2.5, 4.2)
    # plt.xticks(np.arange(0, 65, 5))
    # plt.yticks(np.arange(2.7, 4.3, 0.1))
    # Add ticks on the top of the plot and show it with the layer number
    def layer_number(x):
        return x / 0.7

    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('Layer #', fontweight="bold", size=30,labelpad=18)
    secax.set_xticks(np.arange(0, 91, 10.5))
    secax.set_xticklabels([f'{int(layer_number(x))}' for x in np.arange(0, 91, 10.5)])
    secax.tick_params(axis='both', labelcolor='black', labelsize=30)

    # plt.title('Thin Wall with 1200$^\circ$C Control Method Comparison',fontname="Times New Roman", size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=30)
    plt.xlabel('Height(mm)', fontweight="bold",size=30)
    plt.ylabel('Width(mm)', fontweight="bold",size=30)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower left',ncol=2,fontsize=23.5,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,HC],['TW 25 No Control','TW 25 LP Control','TW 25 FR Control','TW 25 HybridControl'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

    # Print the arrays after the loop
    print("No_Ctrl_width array:", no_ctrl_array)
    print("LP_width array:", lp_array)
    print("FR_width array:", fr_array)
    print("HC_width array:", hc_array)

    # Calculate and print the average and standard deviation for each array
    no_ctrl_mean = np.mean(no_ctrl_array)
    no_ctrl_std = np.std(no_ctrl_array)
    lp_mean = np.mean(lp_array)
    lp_std = np.std(lp_array)
    fr_mean = np.mean(fr_array)
    fr_std = np.std(fr_array)
    hc_mean = np.mean(hc_array)
    hc_std = np.std(hc_array)

    print(f"No_Ctrl_width mean: {no_ctrl_mean}, std: {no_ctrl_std}")
    print(f"LP_width mean: {lp_mean}, std: {lp_std}")
    print(f"FR_width mean: {fr_mean}, std: {fr_std}")
    print(f"HC_width mean: {hc_mean}, std: {hc_std}")
    plt.tight_layout()
    plt.rc('font', weight='bold')
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.73, bottom=0.12)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
    
#*************************Plot section Plot***************************************************   
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(No_Ctrl_df[0]-(No_Ctrl_mid_x-LP_mid_x), No_Ctrl_df[1],color='black', s=5, marker='.',label='NO Ctrl')  
    ax.scatter(LP_df[0], LP_df[1],color=color_lp, s=5, marker='.',label='LP Ctrl')
    ax.scatter(FR_df[0]-(FR_mid_x-LP_mid_x), FR_df[1],color=color_fr,s=5 ,  marker='.',label='FR Ctrl' )
    ax.scatter(HC_df[0]-(HC_mid_x-LP_mid_x), HC_df[1],color='tab:blue',s=5 ,  marker='.',label='Hybrid Ctrl' )
    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=23.5,markerscale = 5)
    # plt.title('Thin Wall Side Profile',fontname="Times New Roman", size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=30)
    plt.xlabel('Width-X coordiante (mm)', fontweight="bold",size=30)
    plt.ylabel('Height-Z coordinate(mm)', fontweight="bold",size=30)
    plt.xlim(1, 5.8)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.12)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
    plt.rc('font', weight='bold')

#*************************Line in section Plot 1100 and 1200 Comparison***************************************************    
plt.show()
# %%
