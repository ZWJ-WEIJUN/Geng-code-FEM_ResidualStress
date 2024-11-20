
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
    cleaned_out_lines = []
    Invalidlines = []
    Index =[]
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
    
    # Assume we are working with the first and the second invalid line (indices 0 and 13149)
    for i in range (len(cleaned_out_lines)):
        Index.append(cleaned_out_lines[i][0])
    
    NC =  lines[Index[0]:Index[1]-1]+lines[Index[1]:Index[2]-1]+lines[Index[2]:Index[3]-1]+lines[Index[3]:Index[4]-1]+lines[Index[4]:Index[5]-1]
    LP =  lines[Index[5]:Index[6]-1]+lines[Index[6]:Index[7]-1]+lines[Index[7]:Index[8]-1]+lines[Index[8]:Index[9]-1]+lines[Index[9]:Index[10]-1]
    FR =  lines[Index[10]:Index[11]-1]+lines[Index[11]:Index[12]-1]+lines[Index[12]:Index[13]-1]+lines[Index[13]:Index[14]-1]+lines[Index[14]:Index[15]-1]
    HC =  lines[Index[15]:Index[16]-1]+lines[Index[16]:Index[17]-1]+lines[Index[17]:Index[18]-1]+lines[Index[18]:Index[19]-1]+lines[Index[19]:Index[20]-1]
    # Create new variables to store the cleaned data
    
    return NC, LP, FR, HC



def width_cal(cleanData,layer_target,y_range,y_inc):
    
    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
    df = pd.DataFrame([x.split(',') for x in cleanData]) # Convert the data into a DataFrame
    df = df.map(str.strip)  # Remove leading and trailing whitespace characters
    

    df = df.astype(float)  # Convert the data into float

    df.iloc[:, 1] = df.iloc[:, 1].abs()
    # df.loc[df[1]>42, df[0]]+=1
    df.iloc[df[1].gt(42).values, 0] += 0.3
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

    
    left_x_mean = []
    right_x_mean = []
    left_x_SD =[]
    right_x_SD = []
    Height_info = []
    
    for i in layer_target:
        for j in np.arange(i, i+y_range,y_inc):
            # cal_left = df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)][1]
            # print(len(cal_left))
            Height_info.append(j)
            cal_left = df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)][0].mean()
            left_x_mean.append(cal_left)
            left_x_SD.append( df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)][0].std())
            # cal_right = df_right[(df_right[1] > j) & (df_right[1] < j+1)][1]
            # right_x_mean.append(cal_right)
            cal_right = df_right[(df_right[1] > j) & (df_right[1] < j+y_inc)][0].mean()
            right_x_mean.append(cal_right)
            right_x_SD.append( df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)][0].std())

            
    
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
    y_range= 15
    y_inc = 1
    Height_target= [3,24,45] 
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    expected_number_of_fields = 3
    
    # file_path_NC_STW65 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW65mm_and_No_Control_050324.csv'
    # file_path_LP_FR = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_Spline.csv'
    # file_path_Comb_1 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/Combined_I_1100_and_1200.csv'
    # file_path_Comb_2 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/Combine_II_1200.csv'
    # file_path_LP_FR_1100 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_1100.csv'
    
        
    # file_path = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Profile_Data_Analysis/Width_Data_Analysis/STW_4xThicknessProfile(H,FR,LP,No).csv'
    file_path = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/4x_STW_ThicknessProfileRedo(No,LP,FR,H).csv'
    # file_path = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_4x_STW_ThicknessProfileRedo(No,LP,FR,H)_09262024.csv'

    STW70_No_Ctrl, STW70_LP,STW70_FR,Combined_II_1200  = clean_data_v1(file_path, expected_number_of_fields)

    

    No_Ctrl_width,Height_info,No_Ctrl_mean,No_Ctrl_SD,No_Ctrl_df, No_Ctrl_mid_x = width_cal(STW70_No_Ctrl,Height_target,y_range,y_inc)
    LP_width,Height_info,LP_mean,LP_SD,LP_df,LP_mid_x = width_cal(STW70_LP,Height_target,y_range,y_inc)
    FR_width,Height_info,FR_mean,FR_SD,FR_df,FR_mid_x = width_cal(STW70_FR,Height_target,y_range,y_inc)
    Combined_II_1200_width,Height_info, Combined_II_1200_mean, Combined_II_1200_SD,Combined_II_1200_df,Combined_II_1200_mid_x = width_cal( Combined_II_1200,Height_target,y_range,y_inc)
 
    print('No Ctrl Mean:', No_Ctrl_mean)
    print('No Ctrl SD:', No_Ctrl_SD)
    print('LP 1200 Mean:', LP_mean)
    print('LP 1200 SD:', LP_SD)
    print('FR 1200 Mean:',FR_mean)
    print('FR 1200 SD:', FR_SD)
    print('Hybrid 1200 Mean:', Combined_II_1200_mean)
    print('Hybrid 1200 SD:', Combined_II_1200_SD)

    Width_Mean= np.array([No_Ctrl_mean,LP_mean,FR_mean,Combined_II_1200_mean])
    Width_SD= np.array([No_Ctrl_SD,LP_SD,FR_SD,Combined_II_1200_SD])  
# %% **********************Comparison    

#**************************Scatter Plot***************************************************     
    fig, ax = plt.subplots(figsize=(10, 8))  
    No_Ctrl = ax.scatter(Height_info,No_Ctrl_width, color='black',label='STW 70 No Control')
    Lp = ax.scatter(Height_info,LP_width, color=color_lp, label='STW 70 LP Control')
    Fr = ax.scatter(Height_info,FR_width, color=color_fr, label='STW 70 FR Control')
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
    Combined_II_1200 = ax.scatter(Height_info,Combined_II_1200_width, color='tab:blue', label='STW 70 Hybrid Control')
    
    plt.ylim(2.4,4.0)
    plt.title('Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)

    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

#*************************Line Plot*************************************************** 
    fig, ax = plt.subplots(figsize=(10, 8))
    No_Ctrl = ax.plot(Height_info,No_Ctrl_width, color='black',label='STW 70 No Control',linewidth=3, marker='.', markersize=17, linestyle='-')
    Lp = ax.plot(Height_info,LP_width, color=color_lp, label='STW 70 LP Control',linewidth=3, marker='.', markersize=17, linestyle='-')
    Fr = ax.plot(Height_info,FR_width, color=color_fr, label='STW 70 FR Control',linewidth=3, marker='.', markersize=17, linestyle='-')
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
    Combined_II_1200 = ax.plot(Height_info,Combined_II_1200_width, color='tab:blue', label='STW 70 Hybrid Control',linewidth=3, marker='.', markersize=17, linestyle='-')

    plt.ylim(2.4,4.0)
    plt.title('1100$^\circ$C Objective Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    
    
 #*************************Line in section Plot STW 70 Comparison***************************************************    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        # No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='STW 70 No Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        # Lp = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_width[i*y_range:(i+1)*y_range], color=color_lp, label='STW 70 LP Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        # Fr = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 70 FR Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        # # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
        # Combined_II_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],Combined_II_1200_width[i*y_range:(i+1)*y_range], color='tab:blue', label='STW 70 Hybrid Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range], No_Ctrl_width[i*y_range:(i+1)*y_range], color='black', label='No Ctrl', linewidth=3, marker='.', markersize=17, linestyle='-')
        Lp = ax.plot(Height_info[i*y_range:(i+1)*y_range], LP_width[i*y_range:(i+1)*y_range], color=color_lp, label='LP Ctrl', linewidth=3, marker='.', markersize=17, linestyle='-')
        Fr = ax.plot(Height_info[i*y_range:(i+1)*y_range], FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='FR Ctrl', linewidth=3, marker='.', markersize=17, linestyle='-')
        Combined_II_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range], Combined_II_1200_width[i*y_range:(i+1)*y_range], color='tab:blue', label='Hybrid Ctrl', linewidth=3, marker='.', markersize=17, linestyle='-')
        
    plt.ylim(2.4,4.0)
    # plt.title('1200$^\circ$C Control Method Comparison',fontname="Times New Roman", size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=2,fontsize=17,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Hybrid Control'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    plt.xlim(0, 65)
    plt.ylim(2.7, 4.2)
    plt.xticks(np.arange(0, 65, 5))
    plt.yticks(np.arange(2.7, 4.3, 0.1))
    # Add ticks on the top of the plot and show it with the layer number
    def layer_number(x):
        return x / 0.7
 
    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('Layer #', fontweight="bold", size=17)
    secax.set_xticks(np.arange(0, 91, 10))
    secax.set_xticklabels([f'{int(layer_number(x))}' for x in np.arange(0, 91, 10)])
    secax.tick_params(axis='both', labelcolor='black', labelsize=17)
 
    plt.axvspan(0*0.7, 30*0.7, facecolor='darkgrey', alpha=0.1)
    plt.axvspan(30*0.7, 60*0.7, facecolor='darkgrey', alpha=0.3)
    plt.axvspan(60*0.7, 90*0.7, facecolor='darkgrey', alpha=0.5)
 



#*************************Plot section Plot***************************************************   
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(No_Ctrl_df[0]-8-(No_Ctrl_mid_x-LP_mid_x), No_Ctrl_df[1],color='black', s=5, marker='.',label='NO Ctrl')  
    ax.scatter(LP_df[0]-8, LP_df[1],color=color_lp, s=5, marker='.',label='LP Ctrl')
    ax.scatter(FR_df[0]-8-(FR_mid_x-LP_mid_x), FR_df[1],color=color_fr,s=5 ,  marker='.',label='FR Ctrl' )
    ax.scatter(Combined_II_1200_df[0]-8-(Combined_II_1200_mid_x-LP_mid_x), Combined_II_1200_df[1],color='tab:blue',s=5 ,  marker='.',label='Hybrid Ctrl')

    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 2.4)
    # plt.title('Step Thin Wall Side Profile',fontname="Times New Roman", size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Width-X coordinate (mm)', fontweight="bold",size=17)
    plt.ylabel('Height-Z coordinate (mm)', fontweight="bold",size=17)
    plt.xticks(np.arange(2, 9, 1))
    plt.xlim(2, 8)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(LP_df[0],LP_df[1])
#*************************Line in section Plot 1100 and 1200 Comparison***************************************************    
    plt.show()