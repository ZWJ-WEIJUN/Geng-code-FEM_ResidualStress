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
    print(df)
    
    fig1, ax1 = plt.subplots()
    # Create a 2D plot
    ax1.scatter(df[0], df[1])

    plt.xlabel('X')
    plt.ylabel('Y')

    # plt.show()
    y_min = 3
    y_max = 60
    
    # y_range = 15
    # y_inc = 0.5
    
    df =df[(df[1] > y_min) & (df[1] < y_max)]
    
    left_x = df[0].min()
    right_x =df[0].max()
    
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
    return Width,Height_info, Width_mean, Width_SD


if __name__ == '__main__':
    # Call the clean_data function
    y_range= 15
    y_inc =1 
    Height_target= [3,24,45] 
    
    file_path = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/STW65mm_and_No_Control_050324.csv'
    expected_number_of_fields = 3
    STW65_FR_Ctrl, STW70_No_Ctrl = clean_data(file_path, expected_number_of_fields)
    
    color_fr = 'tab:purple'
    
    STW65_FR_Ctrl_width,Height_info, STW65_FR_Ctrl_mean, STW65_FR_Ctrl_SD = width_cal( STW65_FR_Ctrl,Height_target,y_range,y_inc)

    print('STW65_FR_CTRL_Mean:', STW65_FR_Ctrl_mean)
    print('STW65_FR_CTRL_SD:', STW65_FR_Ctrl_SD)
    fig2, ax2 = plt.subplots()
    ax2.scatter(Height_info, STW65_FR_Ctrl_width,color=color_fr)
    plt.title('STW65 FR Control')
    plt.xlabel('Height')
    plt.ylabel('Width')
    
    
    STW70_No_Ctrl_width,Height_info,STW70_No_Ctrl_mean,STW70_No_Ctrl_SD = width_cal(STW70_No_Ctrl,Height_target,y_range,y_inc)
    print('STW70_No_Ctrl_Mean:',STW70_No_Ctrl_mean)
    print('STW70_No_Ctrl_SD:',STW70_No_Ctrl_SD)
    
    fig3, ax3 = plt.subplots()
    ax3.scatter(Height_info,STW70_No_Ctrl_width)
    plt.title('STW70 NO Control')
    plt.xlabel('Height')
    plt.ylabel('Width')
    
    