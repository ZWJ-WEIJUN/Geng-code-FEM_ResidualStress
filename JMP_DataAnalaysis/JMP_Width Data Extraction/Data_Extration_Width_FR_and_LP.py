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

def clean_data(file_path, expected_number_of_fields):
    # Variables definition and initialization
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
    section2_lines = lines[second_invalid_index + 1:]
    print(f"Section 1 has {len(section1_lines)} lines")
    print(f"Section 2 has {len(section2_lines)} lines")

    # Create new variables to store the cleaned data
    STW70_LP = section1_lines
    STW70_FR = section2_lines

    return STW70_LP, STW70_FR

def width_cal(cleanData,layer_target,y_range,y_inc):
    
    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
    df = pd.DataFrame([x.split(',') for x in cleanData]) # Convert the data into a DataFrame
    df = df.map(str.strip)  # Remove leading and trailing whitespace characters
    

    df = df.astype(float)  # Convert the data into float
    print(df)

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
    return Width,Height_info, Width_mean, Width_SD, df, mid_x


if __name__ == '__main__':
    # Call the clean_data function
    y_range= 15
    y_inc = 1
    Height_target= [3,24,45] 
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    
    
    file_path = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW70_LP_and_FR_Ctrl_Spline.csv'
    expected_number_of_fields = 3
    STW70_LP, STW70_FR = clean_data(file_path, expected_number_of_fields)
    
  
    
    LP_width,Height_info,LP_mean,LP_SD,LP_df,LP_mid_x = width_cal(STW70_LP,Height_target,y_range,y_inc)
    FR_width,Height_info,FR_mean,FR_SD,FR_df,FR_mid_x = width_cal(STW70_FR,Height_target,y_range,y_inc)
    X_shift  = abs(FR_mid_x-LP_mid_x)
    
    fig2, ax2 = plt.subplots()
    
    Lp = ax2.scatter(Height_info,LP_width, color=color_lp, label='STW 70 LP Control')
    Fr = ax2.scatter(Height_info,FR_width, color=color_fr, label='STW 70 FR Control')
    
    plt.title('STW70 LP and FR CTRL Width')
    plt.xlabel('Height')
    plt.ylabel('Width')

    plt.legend(scatterpoints=1,loc='lower center',ncol=3,fontsize=8,markerscale = 1.0)
    
    
    fig1, ax1 = plt.subplots()
    ax1.scatter(Height_info,LP_width, color=color_lp)
    plt.title('STW70 LP CTRL')
    plt.xlabel('Height')
    plt.ylabel('Width')
    
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_width[i*y_range:(i+1)*y_range], color=color_lp,label='STW 70 No Control',linewidth=3, marker='.', markersize=10, linestyle='-')

    plt.ylim(2.5,3.8)
    plt.title('LP Control Step Thin Wall with 1200$^\circ$C Objective')
    plt.xlabel('Height(mm)')
    plt.ylabel('Width(mm)')
    # plt.legend(scatterpoints=1,loc='lower center',ncol=2,fontsize=8,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=2,fontsize=8,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=2,fontsize=8,markerscale = 1.0)
    
    
    fig1, ax1 = plt.subplots()
    ax1.scatter(Height_info,FR_width, color=color_fr)
    plt.title('STW70 FR CTRL')
    plt.xlabel('Height')
    plt.ylabel('Width')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)):    
        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],LP_width[i*y_range:(i+1)*y_range], color=color_fr,label='STW 70 No Control',linewidth=3, marker='.', markersize=10, linestyle='-')

    plt.ylim(2.5,3.8)
    plt.title('FR Control Step Thin Wall with 1200$^\circ$C Objective')
    plt.xlabel('Height(mm)')
    plt.ylabel('Width(mm)')
    # plt.legend(scatterpoints=1,loc='lower center',ncol=2,fontsize=8,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower center',ncol=2,fontsize=8,markerscale = 1.0)
    # plt.legend([No_Ctrl,Lp,Fr,Combined_II_1200],['STW 70 No Control','STW 70 LP Control','STW 70 FR Control','STW 70 Combined II Control'],scatterpoints=1,loc='lower center',ncol=2,fontsize=8,markerscale = 1.0)
    
    