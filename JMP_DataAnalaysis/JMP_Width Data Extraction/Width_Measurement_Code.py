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
    # For sample without left and right definition and three width measurement 
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

    # Assume we are working with the first and the second invalid line (indices 0 and 13149)
    first_invalid_index = cleaned_out_lines[0][0] - 1  # line 1 (index 0)
    second_invalid_index = cleaned_out_lines[1][0] - 1  # line 13150 (index 13149)
    print(f"First invalid index: {first_invalid_index}")
    print(f"Second invalid index: {second_invalid_index}")

    # Split the data into two sections based on invalid lines
    section1_lines = lines[first_invalid_index + 1:second_invalid_index]
    section2_lines = lines[second_invalid_index + 1:-1]
    print(f"Section 1 has {len(section1_lines)} lines")
    print(f"Section 2 has {len(section2_lines)} lines")

    # Create new variables to store the cleaned data
    STW70_LP = section1_lines
    STW70_FR = section2_lines

    return STW70_LP, STW70_FR


def clean_data_v2(file_path, expected_number_of_fields):
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
    section1_lines = lines[cleaned_out_lines[0][0]:cleaned_out_lines[1][0]-1]+lines[cleaned_out_lines[1][0]:cleaned_out_lines[2][0]-1]

    print(f"Section 1 has {len(section1_lines)} lines")
    # print(f"Section 2 has {len(section2_lines)} lines")

    # Create new variables to store the cleaned data
    Left_Wall = section1_lines
    # Right_Wall = section2_lines

    return Left_Wall


def clean_data_v3(file_path, expected_number_of_fields):
    # Variables definition and initialization
    #For two walls with left and right and three width measurement 
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
    # # Assume we are working with the first and the second invalid line (indices 0 and 13149)
    # first_invalid_index = cleaned_out_lines[0][0] - 1  
    # second_invalid_index = cleaned_out_lines[1][0] - 1 
    # third_invalid_index = cleaned_out_lines[2][0] - 1 
    # fourth_invalid_index = cleaned_out_lines[3][0] - 1 
    # print(f"First invalid index: {first_invalid_index}")
    # print(f"Second invalid index: {second_invalid_index}")
    
    # Split the data into two sections based on invalid lines
    section1_lines = lines[cleaned_out_lines[0][0]:cleaned_out_lines[1][0]-1]+lines[cleaned_out_lines[1][0]:cleaned_out_lines[2][0]-1]
    section2_lines = lines[cleaned_out_lines[-3][0] :cleaned_out_lines[-2][0]-1]+lines[cleaned_out_lines[-2][0]:cleaned_out_lines[-1][0]-1]
    print(f"Section 1 has {len(section1_lines)} lines")
    print(f"Section 2 has {len(section2_lines)} lines")

    
    # Create new variables to store the cleaned data
    Left_Wall = section1_lines
    Right_Wall = section2_lines

    return Left_Wall, Right_Wall

def width_cal(cleanData,layer_target,y_range,y_inc):
    
    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
    df = pd.DataFrame([x.split(',') for x in cleanData]) # Convert the data into a DataFrame
    df = df.map(str.strip)  # Remove leading and trailing whitespace characters
    

    df = df.astype(float)  # Convert the data into float
    # print(df)

    df_original = df[df[0]<83]
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

if __name__ == '__main__':
    # Call the clean_data function
    y_range= 15
    y_inc = 1
    Height_target= [3,24,45] 
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    expected_number_of_fields = 3
    
    file_path_NC_STW65 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW65mm_and_No_Control_050324.csv'
    file_path_LP_FR = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_Spline.csv'
    file_path_Comb_1 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/Combined_I_1100_and_1200.csv'
    file_path_Comb_2 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/Combine_II_1200.csv'
    file_path_LP_FR_1100 = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_1100.csv'
    
        
    # file_path_NC_STW65 = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW65mm_and_No_Control_050324.csv'
    # file_path_LP_FR = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_Spline.csv'
    # file_path_Comb_1 = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/Combined_I_1100_and_1200.csv'
    # file_path_Comb_2 = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/Combine_II_1200.csv'
    # file_path_LP_FR_1100 = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_1100.csv'



    STW65_FR_Ctrl, STW70_No_Ctrl = clean_data_v3(file_path_NC_STW65, expected_number_of_fields)
    STW70_LP, STW70_FR = clean_data_v1(file_path_LP_FR, expected_number_of_fields)
    Combined_I_1100, Combined_I_1200 = clean_data_v3(file_path_Comb_1, expected_number_of_fields)    
    Combined_II_1200= clean_data_v2(file_path_Comb_2, expected_number_of_fields)
    STW70_LP_1100, STW70_FR_1100= clean_data_v1(file_path_LP_FR_1100, expected_number_of_fields)
    

    No_Ctrl_width,Height_info,No_Ctrl_mean,No_Ctrl_SD,No_Ctrl_df, No_Ctrl_mid_x = width_cal(STW70_No_Ctrl,Height_target,y_range,y_inc)
    STW65_FR_width,Height_info, STW65_FR_mean, STW65_FR_SD,STW65_FR_df, STW65_FR_mid_x = width_cal( STW65_FR_Ctrl,Height_target,y_range,y_inc)
    LP_width,Height_info,LP_mean,LP_SD,LP_df,LP_mid_x = width_cal(STW70_LP,Height_target,y_range,y_inc)
    FR_width,Height_info,FR_mean,FR_SD,FR_df,FR_mid_x = width_cal(STW70_FR,Height_target,y_range,y_inc)
    Combined_I_1100_width,Height_info, Combined_I_1100_mean, Combined_I_1100_SD,Combined_I_1100_df,Combined_I_1100_mid_x = width_cal( Combined_I_1100,Height_target,y_range,y_inc)
    Combined_I_1200_width,Height_info,Combined_I_1200_mean,Combined_I_1200_SD,Combined_I_1200_df,Combined_I_1200_mid_x = width_cal(Combined_I_1200,Height_target,y_range,y_inc)
    Combined_II_1200_width,Height_info, Combined_II_1200_mean, Combined_II_1200_SD,Combined_II_1200_df,Combined_II_1200_mid_x = width_cal( Combined_II_1200,Height_target,y_range,y_inc)
    LP_1100_width,Height_info,LP_1100_mean,LP_1100_SD,LP_1100_df,LP_1100_mid_x = width_cal(STW70_LP_1100,Height_target,y_range,y_inc)
    FR_1100_width,Height_info,FR_1100_mean,FR_1100_SD,FR_1100_df,FR_1100_mid_x = width_cal(STW70_FR_1100,Height_target,y_range,y_inc)
    
    print('No Ctrl Mean:', No_Ctrl_mean)
    print('No Ctrl SD:', No_Ctrl_SD)
    print('LP 1100 Mean:', LP_1100_mean)
    print('LP 1100 SD:', LP_1100_SD)
    print('FR 1100 Mean:', FR_1100_mean)
    print('FR 1100 SD:', FR_1100_SD)
    print('Combined I 1100 Mean:', Combined_I_1100_mean)
    print('Combined I 1100 SD:', Combined_I_1100_SD)
    print('LP 1200 Mean:', LP_mean)
    print('LP 1200 SD:', LP_SD)
    print('FR 1200 Mean:',FR_mean)
    print('FR 1200 SD:', FR_SD)
    print('Combined I 1200 Mean:', Combined_I_1200_mean)
    print('Combined I 1200 SD:', Combined_I_1200_SD)
    print('Combined II 1200 Mean:', Combined_II_1200_mean)
    print('Combined II 1200 SD:', Combined_II_1200_SD)
    print('STW 65 1200 Mean:', STW65_FR_mean)
    print('STW65 1200 SD:', STW65_FR_SD)
    
    
# %% **********************Comparison    

#**************************Scatter Plot***************************************************     
    fig, ax = plt.subplots(figsize=(10, 8))  
    No_Ctrl = ax.scatter(Height_info,No_Ctrl_width, color='black',label='STW 70 No Control')
    Lp = ax.scatter(Height_info,LP_width, color=color_lp, label='STW 70 LP Control')
    Fr = ax.scatter(Height_info,FR_width, color=color_fr, label='STW 70 FR Control')
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
    Combined_II_1200 = ax.scatter(Height_info,Combined_II_1200_width, color='tab:blue', label='STW 70 Combined II Control')
    
    plt.ylim(2.4,3.8)
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
    Combined_II_1200 = ax.plot(Height_info,Combined_II_1200_width, color='tab:blue', label='STW 70 Combined II Control',linewidth=3, marker='.', markersize=17, linestyle='-')

    plt.ylim(2.4,3.8)
    plt.title('1100$^\circ$C Objective Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    
    
 #*************************Line in section Plot STW 70 Comparison***************************************************    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='STW 70 No Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        Lp = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_width[i*y_range:(i+1)*y_range], color=color_lp, label='STW 70 LP Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        Fr = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 70 FR Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
        Combined_II_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],Combined_II_1200_width[i*y_range:(i+1)*y_range], color='tab:blue', label='STW 70 Combined II Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
    
    plt.ylim(2.4,3.8)
    plt.title('1200$^\circ$C Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

#*************************Plot section Plot***************************************************   
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(No_Ctrl_df[0]-(No_Ctrl_mid_x-LP_mid_x), No_Ctrl_df[1],color='black', s=5, marker='.',label='STW 70 NO Control')  
    ax.scatter(LP_df[0], LP_df[1],color=color_lp, s=5, marker='.',label='STW 70 LP Control')
    ax.scatter(FR_df[0]-(FR_mid_x-LP_mid_x), FR_df[1],color=color_fr,s=5 ,  marker='.',label='STW 70 FR Control' )
    ax.scatter(Combined_II_1200_df[0]-(Combined_II_1200_mid_x-LP_mid_x), Combined_II_1200_df[1],color='tab:blue',s=5 ,  marker='.',label='STW 70 Combined II Control')
    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 2.4)
    plt.title('Step Thin Wall Side Profile',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Position x (mm)', fontweight="bold",size=17)
    plt.ylabel('Position y (mm)', fontweight="bold",size=17)
    plt.xlim(8.5,14.5)

#*************************Line in section Plot 1100 and 1200 Comparison***************************************************    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='STW 70 No Control',linewidth=3, marker='.', markersize=17, linestyle='-')
        Lp_1100 = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_1100_width[i*y_range:(i+1)*y_range], color=color_lp, label='STW 70 LP Control with 1100$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        Fr_1100 = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_1100_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 70 FR Control with 1100$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
        Combined_I_1100 = ax.plot(Height_info[i*y_range:(i+1)*y_range],Combined_I_1100_width[i*y_range:(i+1)*y_range], color='tab:blue', label='STW 70 Combined I Control with 1100$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
    
    plt.ylim(2.4,3.8)
    plt.title('1100$^\circ$C Objective Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='STW 70 No Control',linewidth=3, marker='.', markersize=17, linestyle='-')
        Combined_I_1100 = ax.plot(Height_info[i*y_range:(i+1)*y_range],Combined_I_1100_width[i*y_range:(i+1)*y_range], color='tab:blue', label='STW 70 Combined I Control with 1100$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        Combined_I_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],Combined_I_1200_width[i*y_range:(i+1)*y_range], color='blue', label='STW 70 Combined I Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        Combined_II_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],Combined_II_1200_width[i*y_range:(i+1)*y_range], color='cornflowerblue', label='STW 70 Combined II Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')

    plt.ylim(2.4,3.8)
    plt.title('Combined Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='STW 70 No Control',linewidth=3, marker='.', markersize=17, linestyle='-')
        Lp_1100 = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_1100_width[i*y_range:(i+1)*y_range], color=color_lp, label='STW 70 LP Control with 1100$^\circ$C Objective',linewidth=3, marker='s', markersize=7, linestyle='--')
        Fr_1100 = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_1100_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 70 FR Control with 1100$^\circ$C Objective',linewidth=3, marker='s', markersize=7, linestyle='--')
        # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='STW 70 Combined I Control')
        Lp_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_width[i*y_range:(i+1)*y_range], color=color_lp, label='STW 70 LP Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
        Fr_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 70 FR Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
    
    plt.ylim(2.4,3.8)
    plt.title('LP and FR Control Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0, handlelength=4)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    

        STW65_FR_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],STW65_FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 65 FR Control with 1200$^\circ$C Objective',linewidth=3, marker='s', markersize=7, linestyle='--')
        Fr_1200 = ax.plot(Height_info[i*y_range:(i+1)*y_range],FR_width[i*y_range:(i+1)*y_range], color=color_fr, label='STW 70 FR Control with 1200$^\circ$C Objective',linewidth=3, marker='.', markersize=17, linestyle='-')
    
    plt.ylim(2.4,3.8)
    plt.title('FR Verification Control Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height(mm)', fontweight="bold",size=17)
    plt.ylabel('Width(mm)', fontweight="bold",size=17)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0, handlelength=4)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

#*************************Line in section Plot 1100 and 1200 Comparison***************************************************   

    fig, ax = plt.subplots(figsize=(10, 8))  
    NC = ax.scatter(No_Ctrl_df[0]-78.5,No_Ctrl_df[1], color='black',label='STW No Control',s=5)
    # plt.ylim(2.4,3.8)
    plt.title('No Control Profile',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Length-Y(mm)', fontweight="bold",size=17)
    plt.ylabel('Height-Z(mm)', fontweight="bold",size=17)
    plt.xlim()
    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)