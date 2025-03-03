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
# plt.close('all')
plt.rcParams["font.family"] = "Times New Roman"

def clean_data_v1(file_path, expected_number_of_fields):
    # Call the clean_data function
    y_range= 15
    y_inc = 1
    Width_target= [3,24,45] 
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    expected_number_of_fields = 3
    
    
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
    
    
    section1_lines = lines[0:cleaned_out_lines[1][0]-2]
    
    print(f"Section 1 has {len(section1_lines)} lines")
    # print(f"Section 2 has {len(section2_lines)} lines")
    
    # Create new variables to store the cleaned data
    Wall = section1_lines
    # Right_Wall = section2_lines

    return Wall


def Height_cal(cleanData,Extraction_Range, x_deducted):
    
    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
     df = pd.DataFrame([x.split(',') for x in cleanData]) # Convert the data into a DataFrame
     df = df.map(str.strip)  # Remove leading and trailing whitespace characters
    
    
     df = df.astype(float)  # Convert the data into float
     df_profile = df[df[1]>-0.5]
     # print(df)

     df_H1 =df[(df[1] > Extraction_Range[0]) & (df[1] < Extraction_Range[1])] #Height 1
     df_H2 =df[(df[1] > Extraction_Range[2]) & (df[1] < Extraction_Range[3])] #Height 2
     df_H3 =df[(df[1] > Extraction_Range[4]) & (df[1] < Extraction_Range[5])] #Height 3
    
     Mid_H1 =  (df_H1[0].min() +df_H1[0].max())/2
     Mid_H2 =  (df_H2[0].min() +df_H2[0].max())/2
     Mid_H3 =  (df_H3[0].min() +df_H3[0].max())/2
    
     x_position = [0,1,2,3,4,5]
     x_position[0] = df_H1[(df_H1[0]<Mid_H1)][0].mean()
     x_position[1] = df_H2[(df_H2[0]<Mid_H2)][0].mean()
     x_position[2] = df_H3[(df_H3[0]<Mid_H3)][0].mean()
     x_position[3] = df_H3[(df_H3[0]>Mid_H3)][0].mean()
     x_position[4] = df_H2[(df_H2[0]>Mid_H2)][0].mean()
     x_position[5] = df_H1[(df_H1[0]>Mid_H1)][0].mean()
    
     df_H1_left =df[(df[0] > x_position[0]+x_deducted) & (df[0] < x_position[1]-x_deducted) & (df[1]>0)] #Height 1
     df_H2_left =df[(df[0] > x_position[1]+x_deducted) & (df[0] < x_position[2]-x_deducted) & (df[1]>0)] #Height 2
     df_H3_mid  =df[(df[0] > x_position[2]+x_deducted) & (df[0] < x_position[3]-x_deducted)  & (df[1]>0)] #Height 3
     df_H2_right =df[(df[0] > x_position[3]+x_deducted) & (df[0] < x_position[4]-x_deducted)& (df[1]>0)] #Height 2
     df_H1_right =df[(df[0] > x_position[4]+x_deducted) & (df[0] < x_position[5]-x_deducted)& (df[1]>0)] #Height 1
    
     df_height_1  = pd.concat([df_H1_left, df_H1_right])
     df_height_2  = pd.concat([df_H2_left, df_H2_right])
     df_height_3  = df_H3_mid        
     H1_mean = df_height_1[1].mean()
     H2_mean = df_height_2[1].mean()
     H3_mean = df_height_3[1].mean()
     H1_sd = df_height_1[1].std()
     H2_sd = df_height_2[1].std()
     H3_sd = df_height_3[1].std()
     STW_mean = [H1_mean,H2_mean,H3_mean]
     STW_sd = [H1_sd,H2_sd,H3_sd]
     df_all_height = pd.concat([df_height_1, df_height_2, df_height_3])
     return df_profile, df_all_height, STW_mean, STW_sd
     

if __name__ == '__main__':
    # Call the clean_data function
    Extraction_Range = [3, 15, 24,36, 45,57] #y postion used to extract mean of x average
    x_deducted= 1
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    expected_number_of_fields = 3
    
    file_path_NC = 'D:/Expriment_Data_Extraction/JMP_Height Data Extraction/STW_NoCtrl.csv'
    file_path_LP = 'D:/Expriment_Data_Extraction/JMP_Height Data Extraction/STW_LPCtrl_1200C.csv'
    file_path_FR = 'D:/Expriment_Data_Extraction/JMP_Height Data Extraction/STW_FRCtrl_1200C.csv'
    file_path_HC = 'D:/Expriment_Data_Extraction/JMP_Height Data Extraction/STW_Hybrid_CTRL_2.csv'
        
    # file_path_NC = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_NoCtrl.csv'
    # file_path_LP = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_LPCtrl_1200C.csv'
    # file_path_FR = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_FRCtrl_1200C.csv'
    # file_path_HC = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_Hybrid_CTRL_2.csv'


    NC_Data = clean_data_v1(file_path_NC, expected_number_of_fields)
    LP_Data = clean_data_v1(file_path_LP, expected_number_of_fields)
    FR_Data = clean_data_v1(file_path_FR, expected_number_of_fields)    
    HC_Data = clean_data_v1(file_path_HC, expected_number_of_fields)


    NC_profile,NC_df, NC_mean, NC_SD = Height_cal(NC_Data,  Extraction_Range,x_deducted)
    LP_profile,LP_df, LP_mean, LP_SD = Height_cal(LP_Data,  Extraction_Range,x_deducted)
    FR_profile,FR_df, FR_mean, FR_SD  = Height_cal(FR_Data,  Extraction_Range,x_deducted)
    HC_profile,HC_df, HC_mean, HC_SD = Height_cal(HC_Data,  Extraction_Range,x_deducted)


    print('No Ctrl Mean:', NC_mean)
    print('No Ctrl SD:', NC_SD)
    print('LP 1200 Mean:', LP_mean)
    print('LP 1200 SD:', LP_SD)
    print('FR 1200 Mean:',FR_mean)
    print('FR 1200 SD:', FR_SD)
    print('Hybrid 1200 Mean:', HC_mean)
    print('Hybrid 1200 SD:', HC_SD)
    
# %% **********************Comparison    

#**************************Scatter Plot of height***************************************************     
    fig, ax = plt.subplots(figsize=(10, 8))  
    NC = ax.scatter(NC_df[0],NC_df[1], color='black',label='STW No Control',s=5)
    Lp = ax.scatter(LP_df[0],LP_df[1], color=color_lp, label='STW LP Control',s=5)
    Fr = ax.scatter(FR_df[0],FR_df[1], color=color_fr, label='STW FR Control',s=5)
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW Combined I Control')
    HC = ax.scatter(HC_df[0],HC_df[1], color='tab:blue', label='STW Hybrid Control',s=5)

    # plt.ylim(2.4,3.8)
    plt.title('Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Length-Y(mm)', fontweight="bold",size=17)
    plt.ylabel('Height-Z(mm)', fontweight="bold",size=17)

    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    
    #**************************Scatter Plot of profile***************************************************     
    fig, ax = plt.subplots(figsize=(10, 8))  
    NC = ax.scatter(NC_profile[0],NC_profile[1], color='black',label='STW No Control',s=5)
    Lp = ax.scatter(LP_profile[0],LP_profile[1], color=color_lp, label='STW LP Control',s=5)
    Fr = ax.scatter(FR_profile[0],FR_profile[1], color=color_fr, label='STW FR Control',s=5)
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW Combined I Control')
    HC = ax.scatter(HC_profile[0],HC_profile[1], color='tab:blue', label='STW Hybrid Control',s=5)
    # plt.ylim(2.4,3.8)
    plt.title('No Control Profile',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Length-Y(mm)', fontweight="bold",size=17)
    plt.ylabel('Height-Z(mm)', fontweight="bold",size=17)

    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

#**************************Bar Plot***************************************************  
    
    fig, ax1 = plt.subplots(figsize=(10, 8)) 
    Method = ["No Ctrl","LP Ctrl", "FR Ctrl", "Hybrid Ctrl"]
    Height_value = [NC_mean[2], LP_mean[2], FR_mean[2], HC_mean[2]]
    Height_SD = [NC_SD[2], LP_SD[2], FR_SD[2], HC_SD[2]]
    Width_value = [3.475,3.274,3.220,3.196]
    Width_SD = [0.242,0.155,0.109,0.125]
    bar_width = 0.4
    x = np.arange(len(Method))
    # Create the figure and axes

    ax2= ax1.twinx()
    

    bar1 = ax1.bar(x - bar_width/2, Height_value,yerr=Height_SD, color ="orange", width = 0.4, label='Height')
    bar2 = ax2.bar(x + bar_width/2, Width_value,yerr=Width_SD, color ="grey", width = 0.4, label ='Width')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(Method)
    ax1.set_ylabel("Height Value(mm)")
    ax2.set_ylabel("Width Value(mm)")
    ax1.set_ylim(40,75)
    ax2.set_ylim(1,4)
    ax1.legend(lines + lines2, labels + labels2, loc='best', ncol=2)
    plt.show()


