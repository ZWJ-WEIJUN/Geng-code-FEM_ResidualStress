# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:22:41 2024

@author: Muqing
"""

import pandas as pd
import matplotlib.pyplot as plt
file_path = 'D:/Expriment_Data_Extraction/JMP_Height Data Extraction/STW_NoCtrl.csv'
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
# *********************************************************************************************************************
# Start of the data analysis
# *********************************************************************************************************************
# Create a 2D plot for STW70_LP
df = pd.DataFrame([x.split(',') for x in Wall]) # Convert the data into a DataFrame
df = df.map(str.strip)  # Remove leading and trailing whitespace characters


df = df.astype(float)  # Convert the data into float
# print(df)
Extraction_Range = [3, 15, 24,36, 45,57]
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

x_deducted = 1
df_H1_left =df[(df[0] > x_position[0]+x_deducted) & (df[0] < x_position[1]-x_deducted) & (df[1]>0)] #Height 1
df_H2_left =df[(df[0] > x_position[1]+x_deducted) & (df[0] < x_position[2]-x_deducted) & (df[1]>0)] #Height 2
df_H3_mid  =df[(df[0] > x_position[2]+x_deducted) & (df[0] < x_position[3]-x_deducted)  & (df[1]>0)] #Height 3
df_H2_right =df[(df[0] > x_position[3]+x_deducted) & (df[0] < x_position[4]-x_deducted)& (df[1]>0)] #Height 2
df_H1_right =df[(df[0] > x_position[4]+x_deducted) & (df[0] < x_position[5]-x_deducted)& (df[1]>0)] #Height 1

df_height_1  = pd.concat([df_H1_left, df_H1_right])
df_height_2  = pd.concat([df_H2_left, df_H2_right])
df_height_3  = df_H3_mid        
Height_1 = df_height_1[1].mean()
Height_2 = df_height_2[1].mean()
Height_3 = df_height_3[1].mean()
Profile = pd.concat([df_height_1, df_height_2, df_height_3])

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(Profile[0], Profile[1])