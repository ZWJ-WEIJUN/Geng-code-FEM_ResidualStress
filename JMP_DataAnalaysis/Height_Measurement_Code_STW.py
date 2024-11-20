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
     


def clean_data_v1_width(file_path, expected_number_of_fields):
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


def clean_data_v2_width(file_path, expected_number_of_fields):
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


def clean_data_v3_width(file_path, expected_number_of_fields):
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
    Extraction_Range = [3, 15, 24,36, 45,57] #y postion used to extract mean of x average
    x_deducted= 1
    color_lp = 'indianred'
    color_fr = 'tab:purple'
    expected_number_of_fields = 3
    

    file_path_NC = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_NoCtrl.csv'
    file_path_LP = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_LPCtrl_1200C.csv'
    file_path_FR = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_FRCtrl_1200C.csv'
    file_path_HC = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_Hybrid_CTRL_2.csv'
        
    # file_path_NC = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_NoCtrl.csv'
    # file_path_LP = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_LPCtrl_1200C.csv'
    # file_path_FR = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_FRCtrl_1200C.csv'
    # file_path_HC = 'C:/Users/muqin/OneDrive/Desktop/Reasearch/Width_Data_Analysis/Width_Data_Analysis/STW_Hybrid_CTRL_2.csv'


    NC = clean_data_v1(file_path_NC, expected_number_of_fields)
    LP = clean_data_v1(file_path_LP, expected_number_of_fields)
    FR = clean_data_v1(file_path_FR, expected_number_of_fields)    
    HC = clean_data_v1(file_path_HC, expected_number_of_fields)


    NC_profile,NC_df, NC_mean, NC_SD = Height_cal(NC,  Extraction_Range,x_deducted)
    LP_profile,LP_df, LP_mean, LP_SD = Height_cal(LP,  Extraction_Range,x_deducted)
    FR_profile,FR_df, FR_mean, FR_SD  = Height_cal(FR,  Extraction_Range,x_deducted)
    HC_profile,HC_df, HC_mean, HC_SD = Height_cal(HC,  Extraction_Range,x_deducted)


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
    # fig, ax = plt.subplots(figsize=(10, 8))  
    # NC = ax.scatter(NC_df[0],NC_df[1], color='black',label='STW No Control',s=5)
    # Lp = ax.scatter(LP_df[0],LP_df[1], color=color_lp, label='STW LP Control',s=5)
    # Fr = ax.scatter(FR_df[0],FR_df[1], color=color_fr, label='STW FR Control',s=5)
    # # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW Combined I Control')
    # HC = ax.scatter(HC_df[0],HC_df[1], color='tab:blue', label='STW Hybrid Control',s=5)

    # # plt.ylim(2.4,3.8)
    # plt.title('Control Method Comparison',fontname="Times New Roman",
    #           size=17, fontweight="bold")
    # ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    # plt.xlabel('Length-Y(mm)', fontweight="bold",size=17)
    # plt.ylabel('Height-Z(mm)', fontweight="bold",size=17)

    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    

    import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# Load the first image
image_path_1 = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_NoCtrl.jpg'
img1 = mpimg.imread(image_path_1)

# Load the second image
image_path_2 = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_NoCtrl_Side.jpg'
img2 = mpimg.imread(image_path_2)


# Load the third image -LP_Ctrl_STW
image_path_3 = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW_LPCtrl.jpg'
img3 = mpimg.imread(image_path_3)

# Define the conversion factor: 1 mm = 82 pixels
pixels_per_mm = 82

# Get dimensions of the first image
image1_width_px = img1.shape[1]  # Image width in pixels
image1_height_px = img1.shape[0]  # Image height in pixels
print(f'Image 1 width: {image1_width_px} px, Image 1 height: {image1_height_px} px')

# Get dimensions of the second image
image2_width_px = img2.shape[1]  # Image width in pixels
image2_height_px = img2.shape[0]  # Image height in pixels
print(f'Image 2 width: {image2_width_px} px, Image 2 height: {image2_height_px} px')

# Get dimensions of the third image
image3_width_px = img3.shape[1]  # Image width in pixels
image3_height_px = img3.shape[0]  # Image height in pixels

# Convert pixel dimensions to mm
image1_width_mm = image1_width_px / pixels_per_mm
image1_height_mm = image1_height_px / pixels_per_mm
image2_width_mm = image2_width_px / pixels_per_mm
image2_height_mm = image2_height_px / pixels_per_mm
print(f'Converted Image 1 width: {image1_width_mm:.2f} mm, Converted Image 1 height: {image1_height_mm:.2f} mm')
print(f'Converted Image 2 width: {image2_width_mm:.2f} mm, Converted Image 2 height: {image2_height_mm:.2f} mm')
image3_width_mm = image3_width_px / pixels_per_mm
image3_height_mm = image3_height_px / pixels_per_mm

# Flip the first image data vertically (180-degree flip)
img1_flipped = np.fliplr(img1)

# Rotate the second image data 180 degrees (clockwise)
img2_rotated = np.fliplr(np.rot90(img2, 2))

# Create the first figure for ax1 and ax3
fig1, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_aspect('equal')
ax3.set_aspect('equal')

# Create the second figure for ax2 and ax4
fig2, (ax2, ax4) = plt.subplots(1, 2, figsize=(12, 6))
ax2.set_aspect('equal')
ax4.set_aspect('equal')

# Create the third figure for ax5
fig3, ax5 = plt.subplots(figsize=(6, 6))
ax5.set_aspect('equal')

# Ensure the aspect ratio is the same for both plots to align heights
ax1.set_aspect(aspect='auto', adjustable='box')
ax3.set_aspect(aspect='auto', adjustable='box')

# Display the first image in the first subplot
# ax1.imshow(img1, cmap='gray', origin='lower', extent=[0, image1_width_mm, 0, image1_height_mm])
ax1.imshow(img1_flipped, cmap='gray')
ax1.set_xlim([0, 85])
ax1.set_ylim([0, 65])

ax1.set_xlabel('Length-Y (mm)', fontsize=17, fontweight='bold')
ax1.set_ylabel('Height-Z (mm)', fontsize=17, fontweight='bold')
ax1.set_aspect(aspect='equal')


# Set tick intervals in millimeters for the first subplot
x_ticks_mm_1 = [i for i in range(0, int(image1_width_mm)+1, 10)]  # Setting x-ticks at every 10 mm
y_ticks_mm_1 = [i for i in range(0, 66, 5)]  # Setting y-ticks at every 10 mm
ax1.tick_params(axis='both', labelcolor='black',labelsize=17)

# Set the tick labels based on mm for the first subplot
ax1.set_xticks([x * pixels_per_mm for x in x_ticks_mm_1])
ax1.set_xticklabels([f"{x:.0f}" for x in x_ticks_mm_1], fontsize=17)
ax1.set_yticks([y * pixels_per_mm for y in y_ticks_mm_1])
ax1.set_yticklabels([f"{y:.0f}" for y in y_ticks_mm_1], fontsize=17)

# Add the step thin wall geometry to ax1
step_geometry_x_mm = [x + 1 for x in [5, 5, 15, 15, 25, 25, 55, 55, 65, 65, 75, 75, 85]]
step_geometry_y_mm = [0, 21, 21, 42, 42, 63, 63, 42, 42, 21, 21, 0, 0]
step_geometry_x_px = [x * pixels_per_mm for x in step_geometry_x_mm]
step_geometry_y_px = [y * pixels_per_mm for y in step_geometry_y_mm]
ax1.plot(step_geometry_x_px, step_geometry_y_px, color='grey', linewidth=2)


# Display the second image in the second subplot
ax2.imshow(img2_rotated, cmap='gray')
ax2.set_xlim([0, image2_width_mm])
ax2.set_ylim([0, image2_height_mm])
ax2.set_xlabel('Width-X (mm)', fontsize=17, fontweight='bold')
ax2.set_ylabel('Height-Z (mm)', fontsize=17, fontweight='bold')

# Set tick intervals in millimeters for the second subplot
x_ticks_mm_2 = [i for i in range(0, int(image2_width_mm)+2, 1)]  # Setting x-ticks at every 10 mm
y_ticks_mm_2 = [i for i in range(0, 66, 5)]  # Setting y-ticks at every 10 mm

# Set the tick labels based on mm for the second subplot
ax2.set_xticks([x * pixels_per_mm for x in x_ticks_mm_2])
ax2.set_xticklabels([f"{x:.0f}" for x in x_ticks_mm_2], fontsize=17)
ax2.set_yticks([y * pixels_per_mm for y in y_ticks_mm_2])
ax2.set_yticklabels([f"{y:.0f}" for y in y_ticks_mm_2], fontsize=17)
ax2.tick_params(axis='both', labelcolor='black',labelsize=17)

# Increase the size of the ticks
ax2.tick_params(axis='both', which='major', labelsize=17)

#**************************Scatter Plot of profile***************************************************     
NC = ax3.scatter(NC_profile[0],NC_profile[1], color='black',label='STW No Control',s=5)
ax3.set_xlabel('Length-Y (mm)', fontweight="bold", size=17)
# ax3.set_ylabel('Height-Z (mm)', fontweight="bold", size=17)
# plt.ylim(2.4,3.8)
# plt.title('No Control Profile',fontname="Times New Roman", size=17, fontweight="bold")
ax3.set_aspect(aspect='equal')



# # Set the tick labels based on mm for the third subplot
# Manually set y-ticks from 0 to 65 with intervals of 5
y_ticks = np.arange(0, 66, 5)  # Generate y-ticks from 0 to 65, with step 5
ax3.set_yticks(y_ticks)
ax3.set_yticklabels([f"{int(y)}" for y in y_ticks], fontsize=17)  # Set y-tick labels

# Manually define the x-ticks on ax3 and set them to range from 0 to 80 with intervals of 10
x_ticks_ax3 = [i for i in range(0, 81, 10)]
ax3.set_xticks(x_ticks_ax3)
ax3.set_xticklabels([f"{x}" for x in x_ticks_ax3], fontsize=17)

# # Set the y-ticks for the third subplot to match the first subplot
# y_ticks_mm_3 = [i for i in range(0, 66, 5)]  # Setting y-ticks at every 5 mm
# ax3.set_yticks([y * pixels_per_mm for y in y_ticks_mm_3])
ax3.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=17, fontweight='bold')
ax3.tick_params(axis='both', labelcolor='black', labelsize=17,  labelleft=False)

# # # Set the limits for the third subplot to match the first subplot
ax3.set_xlim([0, 85])
ax3.set_ylim([0, 65])



# Call the clean_data function
y_range= 15
y_inc = 1
Height_target= [3,24,45] 
expected_number_of_fields = 3

file_path_NC_STW65 = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/JMP_DataAnalaysis/STW65mm_and_No_Control_050324.csv'
STW65_FR_Ctrl, STW70_No_Ctrl = clean_data_v3_width(file_path_NC_STW65, expected_number_of_fields)
No_Ctrl_width,Height_info,No_Ctrl_mean,No_Ctrl_SD,No_Ctrl_df, No_Ctrl_mid_x = width_cal(STW70_No_Ctrl,Height_target,y_range,y_inc)
#*************************Single profile plot - No Ctrl***************************************************   
NC = ax4.scatter(No_Ctrl_df[0]-78.5,No_Ctrl_df[1], color='black',label='STW No Control',s=5)
# plt.ylim(2.4,3.8)
# plt.title('No Control Profile',fontname="Times New Roman", size=17, fontweight="bold")
ax4.tick_params(axis='both', labelcolor='black',labelsize=17)
ax4.set_xlabel('Width-X (mm)', fontsize=17, fontweight='bold')
# plt.ylabel('Height-Z(mm)', fontweight="bold",size=17)

# Manually define the y-ticks on ax4 and set them to range from 0 to 65 with intervals of 5
y_ticks_ax4 = [i for i in range(0, 66, 5)]
ax4.set_yticks(y_ticks_ax4)
# ax4.set_yticklabels([f"{y}" for y in y_ticks_ax4], fontsize=17, fontweight='bold')
ax4.tick_params(axis='y', labelleft=False)  # Disable y-axis tick labels


# Manually define the x-ticks on ax4 and set them to range from 0 to 7 with intervals of 1
x_ticks_ax4 = [i for i in range(0, 5, 1)]
ax4.set_xticks(x_ticks_ax4)
ax4.set_xticklabels([f"{x+2}" for x in x_ticks_ax4], fontsize=17)

# Ensure the limits for ax4 are set appropriately
ax4.set_xlim([0, 5])
ax4.set_ylim([0, 65])


# Display the third image in the subplot of fig3
ax5.imshow(img3, cmap='gray')
ax5.set_xlim([0, image3_width_px])
ax5.set_ylim([0, image3_height_px])
ax5.set_xlabel('Length-Y (mm)', fontsize=17, fontweight='bold')
ax5.set_ylabel('Height-Z (mm)', fontsize=17, fontweight='bold')

# Set tick intervals in millimeters for the third subplot
x_ticks_mm_3 = [i for i in range(0, int(image3_width_mm)+1, 10)]  # Setting x-ticks at every 10 mm
y_ticks_mm_3 = [i for i in range(0, 66, 5)]  # Setting y-ticks at every 5 mm
ax5.tick_params(axis='both', labelcolor='black', labelsize=17)

# Set the tick labels based on mm for the third subplot
ax5.set_xticks([x * pixels_per_mm for x in x_ticks_mm_3])
ax5.set_xticklabels([f"{x:.0f}" for x in x_ticks_mm_3], fontsize=17)
ax5.set_yticks([y * pixels_per_mm for y in y_ticks_mm_3])
ax5.set_yticklabels([f"{y:.0f}" for y in y_ticks_mm_3], fontsize=17)

# Add the step thin wall geometry to ax5, shifted by -1mm in the +x direction
step_geometry_x_mm_shifted = [x - 1 for x in [5, 5, 15, 15, 25, 25, 55, 55, 65, 65, 75, 75, 85]]
step_geometry_x_px_shifted = [x * pixels_per_mm for x in step_geometry_x_mm_shifted]
ax5.plot(step_geometry_x_px_shifted, step_geometry_y_px, color='grey', linewidth=2)



# plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

# Adjust layout to reduce space between subplots
plt.subplots_adjust(wspace=0.01)

# # Adjust layout to prevent overlap
# plt.tight_layout()

plt.show()
# %%
