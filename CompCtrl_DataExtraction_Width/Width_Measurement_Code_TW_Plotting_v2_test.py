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
        print(f'Total number of lines: {len(lines)}')
    
    # Function to check if a line contains numerical data
    def is_numerical_line(line):
        try:
            values = line.strip().split(',')
            if len(values) != expected_number_of_fields:
                return False
            # Try to convert all values to float
            for value in values:
                float(value.strip())
            return True
        except (ValueError, AttributeError):
            return False
    
    # Filter out non-numerical lines and identify invalid lines
    valid_lines = []
    for i, line in enumerate(lines):
        if is_numerical_line(line):
            valid_lines.append(line)
        else:
            if line.strip():  # Only add non-empty lines to cleaned_out_lines
                cleaned_out_lines.append((i+1, line))
                print(f"Line {i+1} is not numerical data: {line.strip()}")
    
    print(f"Found {len(valid_lines)} valid numerical data lines")
    print(f"Skipped {len(cleaned_out_lines)} non-numerical lines")
    
    # For this specific case, return all valid numerical lines as Wall data
    Wall = valid_lines
    
    return Wall

def width_cal(cleanData,layer_target,y_range,y_inc):
    
    print(f"width_cal called with {len(cleanData)} data points")
    
    # *********************************************************************************************************************
    # Start of the data analysis
    # *********************************************************************************************************************
    # Create a 2D plot for STW70_LP
    df = pd.DataFrame([x.split(',') for x in cleanData]) # Convert the data into a DataFrame
    
    print(f"DataFrame created with shape: {df.shape}")
    
    df = df.map(str.strip)  # Remove leading and trailing whitespace characters
    

    df = df.astype(float)  # Convert the data into float
    # print(df)
    df.iloc[:, 1] = df.iloc[:, 1].abs()
    df_original = df
    # plt.show()
    y_min = 0.5
    y_max = 55
    
    # y_range = 15
    # y_inc = 0.5
    
    df =df[(df[1] > y_min) & (df[1] < y_max)]
    
    left_x = df[0].min()
    right_x =df[0].max()
    mid_x = (left_x+right_x)/2
    
    df_left = df[(df[0] > left_x-2) & (df[0] < left_x+2)]
    df_right = df[(df[0] > right_x-2) & (df[0] < right_x+2)]
    
    print(f"Total filtered data points: {len(df)}")
    print(f"Left side points: {len(df_left)}")
    print(f"Right side points: {len(df_right)}")
    print(f"Left + Right = {len(df_left) + len(df_right)}")
    print(f"Left X range: {left_x-2} to {left_x+2}")
    print(f"Right X range: {right_x-2} to {right_x+2}")
    
    if (len(df_left)+len(df_right)!=len(df)):
        print(f"WARNING: Data points mismatch! Total: {len(df)}, Left+Right: {len(df_left)+len(df_right)}")
        print("Some points may fall between left and right regions or be duplicated.")
        print("Continuing with available data...")
        # quit()  # Commenting out quit() to continue execution
    
    left_x_mean = []
    right_x_mean = []    
    
    
    Height_info = []
  
    for i in layer_target:
        for j in np.arange(i, i+y_range,y_inc):
            Height_info.append(j)
            
            # Get points in the current height range
            left_points = df_left[(df_left[1] > j) & (df_left[1] < j+y_inc)]
            right_points = df_right[(df_right[1] > j) & (df_right[1] < j+y_inc)]
            
            if len(left_points) == 0:
                cal_left = np.nan
                print(f"Warning: No left points found for height {j:.1f}")
            else:
                cal_left = left_points[0].mean()
            
            if len(right_points) == 0:
                cal_right = np.nan
                print(f"Warning: No right points found for height {j:.1f}")
            else:
                cal_right = right_points[0].mean()
            
            left_x_mean.append(cal_left)
            right_x_mean.append(cal_right)
    
    left_x_mean = np.array(left_x_mean)
    right_x_mean = np.array(right_x_mean)
    Width = right_x_mean - left_x_mean 
    
    # Calculate mean and std ignoring nan values
    Width_mean = np.nanmean(Width)
    Width_SD = np.nanstd(Width)
    
    print(f"Width calculation complete. Valid points: {np.sum(~np.isnan(Width))}/{len(Width)}")
    print(f"Width mean: {Width_mean:.4f}, SD: {Width_SD:.4f}")
    
    return Width,Height_info, Width_mean, Width_SD, df_original, mid_x

def plot_graph(dataframes, mid_x_values, colors, labels):
    """
    Plot multiple dataframes on the same plot
    
    Parameters:
    dataframes: list of dataframes to plot
    mid_x_values: list of mid_x values for x-axis alignment
    colors: list of colors for each dataframe
    labels: list of labels for each dataframe
    """
    fig, ax = plt.subplots(figsize=(12, 10))  
    
    # Use the first dataframe's mid_x as reference for alignment
    reference_mid_x = mid_x_values[0]
    
    for i, (df, mid_x, color, label) in enumerate(zip(dataframes, mid_x_values, colors, labels)):
        # Access dataframe columns by index (0 for x, 1 for y)
        df_x = df.iloc[:, 0]  # First column (x coordinates)
        df_y = df.iloc[:, 1]  # Second column (y coordinates)
        
        # Adjust x coordinates to align all datasets
        x_offset = mid_x - reference_mid_x
        adjusted_x = df_x - x_offset
        
        ax.scatter(adjusted_x, df_y, color=color, label=label, s=5, marker='.')

    plt.title('Thin Wall Side Profile Comparison', fontname="Times New Roman",
              size=18, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black', labelsize=16)
    plt.ylim(0, 65)
    plt.xlabel('Position x (mm)', fontweight="bold", size=16)
    plt.ylabel('Position y (mm)', fontweight="bold", size=16)
    plt.legend(scatterpoints=1, loc='upper right', ncol=1, fontsize=14, markerscale=3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    


if __name__ == '__main__':
    # Call the clean_data function
    y_range= 55
    y_inc = 1.0
    Height_target= [0.5]  # Changed from [3] to [0.5] to start measurements from 0.5mm
    color_pfr = 'grey'
    color_pfr_height = 'tab:blue'
    color_nc = 'black'
    color_pfr_height_energy_lp = 'red'
    color_pfr_height_energy_lp_modified = 'indianred'
    color_pfr_height_energy_lp_1223_1 = 'tab:orange'
    color_comp_w_fr_ctrl = 'tab:purple'
    color_comp_w_energy_ctrl = 'tab:green'
    color_w_fr_ctrl_python_wrong = 'tab:brown'
    color_w_fr_ctrl_lp_max_out = 'tab:pink'
    color_w_fr_ctrl_53layer = 'tab:cyan'
    expected_number_of_fields = 3
    

    
            
    # file_path_NC = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_NoCtrl_ThicknessProfile.csv'
    # file_path_LP = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_LPCtrl_ThicknessProfile.csv'
    # file_path_FR = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_FRCtrl_ThicknessProfile.csv'
    # file_path_HC = 'D:/Expriment_Data_Extraction/Width_Data_Analysis/Width_Data_Analysis/TW25mm_Hybrid3Ctrl_ThicknessProfile.csv'

            
    # file_path_NC = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_NoCtrl_ThicknessProfile.csv'
    # file_path_LP = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_LPCtrl_ThicknessProfile.csv'
    # file_path_FR = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_FRCtrl_ThicknessProfile.csv'
    # file_path_HC = 'C:/Users/muqin/OneDrive/Desktop/Jourmal/Width_Data_Analysis/TW25mm_Hybrid3Ctrl_ThicknessProfile.csv'


    file_path_NC = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/CompCtrl_WP1_#4_CMM_Cam.csv'
    file_path_PFR = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/CompCtrl_WP1_#5_CMM_Cam.csv'
    file_path_PFR_Height = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/CompCtrl_WP2_#6_CMM_Cam.csv'
    # file_path_PFR_Height_EnergyLP = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/CompCtrl_WP3_#3_CMM_Cam.csv'
    file_path_PFR_Height_EnergyLP_Modified = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP4_#2_CMM_Cam-2025-11-12.csv'
    # file_path_PFR_Height_EnergyLP_1223degC = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP4_#1_CMM_Cam-2025-11-14.csv'
    file_path_Comp_W_FR_Ctrl = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP_Raw_#1_Trial_1_40B.csv'
    file_path_Comp_W_Energy_Ctrl = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP5_#3_CMM_Cam_2025-11-25.csv'
    file_path_W_FR_Ctrl_PythonWrong = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP5_#1_CMM_Cam.csv'
    file_path_W_FR_Ctrl_LPMaxOut = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP5_#2_CMM_Cam.csv'
    file_path_W_FR_Ctrl_53Layer = '/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/CompCtrl_DataExtraction_Width/WP_Raw_#3_CMM_Cam_B60_Zoom68X_2025-11-29.csv'




    TW80_NC = clean_data_v1(file_path_NC, expected_number_of_fields)
    TW80_PFR = clean_data_v1(file_path_PFR, expected_number_of_fields)
    TW80_PFR_Height = clean_data_v1(file_path_PFR_Height, expected_number_of_fields)   
    # TW80_PFR_Height_LP = clean_data_v1(file_path_PFR_Height_EnergyLP, expected_number_of_fields)
    TW80_PFR_Height_LP_Modified = clean_data_v1(file_path_PFR_Height_EnergyLP_Modified, expected_number_of_fields)
    # TW80_PFR_Height_LP_1223degC = clean_data_v1(file_path_PFR_Height_EnergyLP_1223degC, expected_number_of_fields)
    TW80_Comp_W_FR_Ctrl = clean_data_v1(file_path_Comp_W_FR_Ctrl, expected_number_of_fields)
    TW80_Comp_W_Energy_Ctrl = clean_data_v1(file_path_Comp_W_Energy_Ctrl, expected_number_of_fields)
    TW80_W_FR_Ctrl_PythonWrong = clean_data_v1(file_path_W_FR_Ctrl_PythonWrong, expected_number_of_fields)
    TW80_W_FR_Ctrl_LPMaxOut = clean_data_v1(file_path_W_FR_Ctrl_LPMaxOut, expected_number_of_fields)
    TW80_W_FR_Ctrl_53Layer = clean_data_v1(file_path_W_FR_Ctrl_53Layer, expected_number_of_fields)

    print('Data Cleaned! Start Width Calculation\n')


    print('Starting width calculation for No_Ctrl...')
    No_Ctrl_width,Height_info,No_Ctrl_mean,No_Ctrl_SD,No_Ctrl_df, No_Ctrl_mid_x = width_cal(TW80_NC,Height_target,y_range,y_inc)
    print('Completed width calculation for No_Ctrl')
    
    print('Starting width calculation for PFR_Ctrl...')
    PFR_Ctrl_width,Height_info,PFR_Ctrl_mean,PFR_Ctrl_SD,PFR_Ctrl_df,PFR_Ctrl_mid_x = width_cal(TW80_PFR,Height_target,y_range,y_inc)
    print('Completed width calculation for PFR_Ctrl')
    
    print('Starting width calculation for PFR_Height_Ctrl...')
    PFR_Height_Ctrl_width,Height_info,PFR_Height_Ctrl_mean,PFR_Height_Ctrl_SD,PFR_Height_Ctrl_df,PFR_Height_Ctrl_mid_x = width_cal(TW80_PFR_Height,Height_target,y_range,y_inc)
    print('Completed width calculation for PFR_Height_Ctrl')
    
    # print('Starting width calculation for PFR_Height_Energy_LP...')
    # PFR_Height_Energy_LP_width,Height_info, PFR_Height_Energy_LP_mean, PFR_Height_Energy_LP_SD,PFR_Height_Energy_LP_df,PFR_Height_Energy_LP_mid_x = width_cal(TW80_PFR_Height_LP,Height_target,y_range,y_inc)
    # print('Completed width calculation for PFR_Height_Energy_LP')
    
    print('Starting width calculation for PFR_Height_Energy_LP_Modified...')
    PFR_Height_Energy_LP_Modified_width,Height_info, PFR_Height_Energy_LP_Modified_mean, PFR_Height_Energy_LP_Modified_SD,PFR_Height_Energy_LP_Modified_df,PFR_Height_Energy_LP_Modified_mid_x = width_cal(TW80_PFR_Height_LP_Modified,Height_target,y_range,y_inc)
    print('Completed width calculation for PFR_Height_Energy_LP_Modified')
    
    # print('Starting width calculation for PFR_Height_Energy_LP_1223degC...')
    # PFR_Height_Energy_LP_1223degC_width,Height_info, PFR_Height_Energy_LP_1223degC_mean, PFR_Height_Energy_LP_1223degC_SD,PFR_Height_Energy_LP_1223degC_df,PFR_Height_Energy_LP_1223degC_mid_x = width_cal(TW80_PFR_Height_LP_1223degC,Height_target,y_range,y_inc)
    # print('Completed width calculation for PFR_Height_Energy_LP_1223degC')
    
    print('Starting width calculation for Comp_W_FR_Ctrl...')
    Comp_W_FR_Ctrl_width,Height_info, Comp_W_FR_Ctrl_mean, Comp_W_FR_Ctrl_SD,Comp_W_FR_Ctrl_df,Comp_W_FR_Ctrl_mid_x = width_cal(TW80_Comp_W_FR_Ctrl,Height_target,y_range,y_inc)
    print('Completed width calculation for Comp_W_FR_Ctrl')
    
    print('Starting width calculation for Comp_W_Energy_Ctrl...')
    Comp_W_Energy_Ctrl_width,Height_info, Comp_W_Energy_Ctrl_mean, Comp_W_Energy_Ctrl_SD,Comp_W_Energy_Ctrl_df,Comp_W_Energy_Ctrl_mid_x = width_cal(TW80_Comp_W_Energy_Ctrl,Height_target,y_range,y_inc)
    print('Completed width calculation for Comp_W_Energy_Ctrl')
    
    print('Starting width calculation for W_FR_Ctrl_PythonWrong...')
    W_FR_Ctrl_PythonWrong_width,Height_info, W_FR_Ctrl_PythonWrong_mean, W_FR_Ctrl_PythonWrong_SD,W_FR_Ctrl_PythonWrong_df,W_FR_Ctrl_PythonWrong_mid_x = width_cal(TW80_W_FR_Ctrl_PythonWrong,Height_target,y_range,y_inc)
    print('Completed width calculation for W_FR_Ctrl_PythonWrong')
    
    print('Starting width calculation for W_FR_Ctrl_LPMaxOut...')
    W_FR_Ctrl_LPMaxOut_width,Height_info, W_FR_Ctrl_LPMaxOut_mean, W_FR_Ctrl_LPMaxOut_SD,W_FR_Ctrl_LPMaxOut_df,W_FR_Ctrl_LPMaxOut_mid_x = width_cal(TW80_W_FR_Ctrl_LPMaxOut,Height_target,y_range,y_inc)
    print('Completed width calculation for W_FR_Ctrl_LPMaxOut')
    
    print('Starting width calculation for W_FR_Ctrl_53Layer...')
    W_FR_Ctrl_53Layer_width,Height_info, W_FR_Ctrl_53Layer_mean, W_FR_Ctrl_53Layer_SD,W_FR_Ctrl_53Layer_df,W_FR_Ctrl_53Layer_mid_x = width_cal(TW80_W_FR_Ctrl_53Layer,Height_target,y_range,y_inc)
    print('Completed width calculation for W_FR_Ctrl_53Layer')

    print('All width calculations completed!')

    
    print('No Ctrl Mean:', No_Ctrl_mean)
    print('No Ctrl SD:', No_Ctrl_SD)
    print('PFR Ctrl 1200 Mean:', PFR_Ctrl_mean)
    print('PFR Ctrl 1200 SD:', PFR_Ctrl_SD)
    print('PFR Height Ctrl 1200 Mean:',PFR_Height_Ctrl_mean)
    print('PFR Height Ctrl 1200 SD:', PFR_Height_Ctrl_SD)

    # print('PFR Height Energy(LP) Mean:', PFR_Height_Energy_LP_mean)
    # print('PFR Height Energy(LP) SD:', PFR_Height_Energy_LP_SD)
    
    print('PFR Height Energy(LP) Modified 1200 Mean:', PFR_Height_Energy_LP_Modified_mean)
    print('PFR Height Energy(LP) Modified 1200 SD:', PFR_Height_Energy_LP_Modified_SD)
    
    # print('PFR Height Energy(LP) 1223degC Mean:', PFR_Height_Energy_LP_1223degC_mean)
    # print('PFR Height Energy(LP) 1223degC SD:', PFR_Height_Energy_LP_1223degC_SD)
    
    print('Comp. W-FR Ctrl Mean:', Comp_W_FR_Ctrl_mean)
    print('Comp. W-FR Ctrl SD:', Comp_W_FR_Ctrl_SD)
    
    print('Comp. W-Energy Ctrl Mean:', Comp_W_Energy_Ctrl_mean)
    print('Comp. W-Energy Ctrl SD:', Comp_W_Energy_Ctrl_SD)
    
    print('W-FR Ctrl_PythonCodeWrong-LPNoChange Mean:', W_FR_Ctrl_PythonWrong_mean)
    print('W-FR Ctrl_PythonCodeWrong-LPNoChange SD:', W_FR_Ctrl_PythonWrong_SD)
    
    print('W-FR Ctrl_LPMaxOut Mean:', W_FR_Ctrl_LPMaxOut_mean)
    print('W-FR Ctrl_LPMaxOut SD:', W_FR_Ctrl_LPMaxOut_SD)
    
    print('W-FR-Ctrl-53Layer Mean:', W_FR_Ctrl_53Layer_mean)
    print('W-FR-Ctrl-53Layer SD:', W_FR_Ctrl_53Layer_SD)
    
# %% **********************Comparison    
    
    # Plot all datasets for side profile comparison
    # dataframes = [No_Ctrl_df, PFR_Ctrl_df, PFR_Height_Ctrl_df, PFR_Height_Energy_LP_df, PFR_Height_Energy_LP_Modified_df, PFR_Height_Energy_LP_1223degC_df, Comp_W_FR_Ctrl_df, Comp_W_Energy_Ctrl_df]
    # mid_x_values = [No_Ctrl_mid_x, PFR_Ctrl_mid_x, PFR_Height_Ctrl_mid_x, PFR_Height_Energy_LP_mid_x, PFR_Height_Energy_LP_Modified_mid_x, PFR_Height_Energy_LP_1223degC_mid_x, Comp_W_FR_Ctrl_mid_x, Comp_W_Energy_Ctrl_mid_x]
    # colors = [color_nc, color_pfr, color_pfr_height, color_pfr_height_energy_lp, color_pfr_height_energy_lp_modified, color_pfr_height_energy_lp_1223_1, color_comp_w_fr_ctrl, color_comp_w_energy_ctrl]
    # labels = ['No Control', 'PFR Ctrl', 'PFR Height Ctrl', 'PFR Height Energy(LP)', 'PFR Height Energy(LP) Modified', 'PFR Height Energy(LP) 1223degC', 'Comp. W-FR Ctrl', 'Comp. W-Energy Ctrl']
    dataframes = [No_Ctrl_df, PFR_Ctrl_df, PFR_Height_Ctrl_df, PFR_Height_Energy_LP_Modified_df, Comp_W_FR_Ctrl_df, Comp_W_Energy_Ctrl_df, W_FR_Ctrl_PythonWrong_df, W_FR_Ctrl_LPMaxOut_df, W_FR_Ctrl_53Layer_df]
    mid_x_values = [No_Ctrl_mid_x, PFR_Ctrl_mid_x, PFR_Height_Ctrl_mid_x, PFR_Height_Energy_LP_Modified_mid_x, Comp_W_FR_Ctrl_mid_x, Comp_W_Energy_Ctrl_mid_x, W_FR_Ctrl_PythonWrong_mid_x, W_FR_Ctrl_LPMaxOut_mid_x, W_FR_Ctrl_53Layer_mid_x]
    colors = [color_nc, color_pfr, color_pfr_height, color_pfr_height_energy_lp_modified, color_comp_w_fr_ctrl, color_comp_w_energy_ctrl, color_w_fr_ctrl_python_wrong, color_w_fr_ctrl_lp_max_out, color_w_fr_ctrl_53layer]
    labels = ['No Ctrl', 'PFR Ctrl', 'QPF Ctrl', 'Comp. T-LP Ctrl', 'Comp. T-FR Ctrl', 'Comp. W-FR Ctrl (WP5 #3)', 'W-FR Ctrl_PythonCodeWrong-LPNoChange (WP5 #1)', 'W-FR Ctrl_LPMaxOut (WP5 #2)', 'W-FR-Ctrl-53Layer (WP_Raw #3)']
    # dataframes = [No_Ctrl_df, PFR_Ctrl_df, PFR_Height_Ctrl_df, PFR_Height_Energy_LP_Modified_df]
    # mid_x_values = [No_Ctrl_mid_x, PFR_Ctrl_mid_x, PFR_Height_Ctrl_mid_x, PFR_Height_Energy_LP_Modified_mid_x]
    # colors = [color_nc, color_pfr, color_pfr_height, color_pfr_height_energy_lp_modified]
    # labels = ['No Ctrl', 'PFR Ctrl', 'QPF Ctrl', 'Comp. LP Ctrl']
    
    plot_graph(dataframes, mid_x_values, colors, labels)

#**************************Scatter Plot***************************************************     
    fig, ax = plt.subplots(figsize=(10, 8))  
    No_Ctrl = ax.scatter(Height_info,No_Ctrl_width, color='black',label='TW 80 No Ctrl')
    PFR_Ctrl = ax.scatter(Height_info,PFR_Ctrl_width, color=color_pfr, label='TW 80 PFR Ctrl')
    PFR_Height_Ctrl = ax.scatter(Height_info,PFR_Height_Ctrl_width, color=color_pfr_height, label='TW 80 QPF Ctrl')
    # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW 25 Combined I Control')
    # PFR_Height_Energy_LP = ax.scatter(Height_info,PFR_Height_Energy_LP_width, color=color_pfr_height_energy_lp, label='TW 80 PFR Height Energy(LP)')
    PFR_Height_Energy_LP_Modified = ax.scatter(Height_info,PFR_Height_Energy_LP_Modified_width, color=color_pfr_height_energy_lp_modified, label='TW 80 Comp. T-LP Ctrl')
    # PFR_Height_Energy_LP_1223degC = ax.scatter(Height_info,PFR_Height_Energy_LP_1223degC_width, color=color_pfr_height_energy_lp_1223_1, label='TW 80 PFR Height Energy(LP) 1223degC')
    Comp_W_FR_Ctrl = ax.scatter(Height_info,Comp_W_FR_Ctrl_width, color=color_comp_w_fr_ctrl, label='TW 80 Comp. T-FR Ctrl')
    Comp_W_Energy_Ctrl = ax.scatter(Height_info,Comp_W_Energy_Ctrl_width, color=color_comp_w_energy_ctrl, label='TW 80 Comp. W-PEF Ctrl')
    W_FR_Ctrl_PythonWrong = ax.scatter(Height_info,W_FR_Ctrl_PythonWrong_width, color=color_w_fr_ctrl_python_wrong, label='TW 80 W-FR Ctrl_PythonCodeWrong-LPNoChange')
    W_FR_Ctrl_LPMaxOut = ax.scatter(Height_info,W_FR_Ctrl_LPMaxOut_width, color=color_w_fr_ctrl_lp_max_out, label='TW 80 W-FR Ctrl_LPMaxOut')
    W_FR_Ctrl_53Layer = ax.scatter(Height_info,W_FR_Ctrl_53Layer_width, color=color_w_fr_ctrl_53layer, label='TW 80 W-FR-Ctrl-53Layer')

    plt.ylim(2.4,4.5)
    plt.title('Control Method Comparison',fontname="Times New Roman",
              size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=17)
    plt.xlabel('Height (mm)', fontweight="bold",size=17)
    plt.ylabel('Width (mm)', fontweight="bold",size=17)

    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

 #*************************Line in section Plot TW 25 Comparison***************************************************    
    # Initialize arrays to store the elements
    no_ctrl_array = []
    pfr_ctrl_array = []
    pfr_height_ctrl_array = []
    # pfr_height_energy_lp_array = []
    pfr_height_energy_lp_modified_array = []
    # pfr_height_energy_lp_1223degC_array = []
    comp_w_fr_ctrl_array = []
    comp_w_energy_ctrl_array = []
    w_fr_ctrl_python_wrong_array = []
    w_fr_ctrl_lp_max_out_array = []
    w_fr_ctrl_53layer_array = []
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(0,len(Height_target)): 
        no_ctrl_segment = No_Ctrl_width[i*y_range:(i+1)*y_range]
        pfr_ctrl_segment = PFR_Ctrl_width[i*y_range:(i+1)*y_range]
        pfr_height_ctrl_segment = PFR_Height_Ctrl_width[i*y_range:(i+1)*y_range]
        # pfr_height_energy_lp_segment = PFR_Height_Energy_LP_width[i*y_range:(i+1)*y_range]
        pfr_height_energy_lp_modified_segment = PFR_Height_Energy_LP_Modified_width[i*y_range:(i+1)*y_range]
        # pfr_height_energy_lp_1223degC_segment = PFR_Height_Energy_LP_1223degC_width[i*y_range:(i+1)*y_range]
        comp_w_fr_ctrl_segment = Comp_W_FR_Ctrl_width[i*y_range:(i+1)*y_range]
        comp_w_energy_ctrl_segment = Comp_W_Energy_Ctrl_width[i*y_range:(i+1)*y_range]
        w_fr_ctrl_python_wrong_segment = W_FR_Ctrl_PythonWrong_width[i*y_range:(i+1)*y_range]
        w_fr_ctrl_lp_max_out_segment = W_FR_Ctrl_LPMaxOut_width[i*y_range:(i+1)*y_range]
        w_fr_ctrl_53layer_segment = W_FR_Ctrl_53Layer_width[i*y_range:(i+1)*y_range]

        # Append segments to arrays
        no_ctrl_array.extend(no_ctrl_segment)
        pfr_ctrl_array.extend(pfr_ctrl_segment)
        pfr_height_ctrl_array.extend(pfr_height_ctrl_segment)
        # pfr_height_energy_lp_array.extend(pfr_height_energy_lp_segment)
        pfr_height_energy_lp_modified_array.extend(pfr_height_energy_lp_modified_segment)
        # pfr_height_energy_lp_1223degC_array.extend(pfr_height_energy_lp_1223degC_segment)
        comp_w_fr_ctrl_array.extend(comp_w_fr_ctrl_segment)
        comp_w_energy_ctrl_array.extend(comp_w_energy_ctrl_segment)
        w_fr_ctrl_python_wrong_array.extend(w_fr_ctrl_python_wrong_segment)
        w_fr_ctrl_lp_max_out_array.extend(w_fr_ctrl_lp_max_out_segment)
        w_fr_ctrl_53layer_array.extend(w_fr_ctrl_53layer_segment)

        No_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],No_Ctrl_width[i*y_range:(i+1)*y_range], color='black',label='No Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        PFR_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],PFR_Ctrl_width[i*y_range:(i+1)*y_range], color=color_pfr, label='PFR Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        PFR_Height_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],PFR_Height_Ctrl_width[i*y_range:(i+1)*y_range], color=color_pfr_height, label='QPF Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        # Combined_I_1200 = ax2.scatter(Height_info,Combined_I_1200_width, color='tab:blue', label='TW 25 Combined I Control')
        # PFR_Height_Energy_LP = ax.plot(Height_info[i*y_range:(i+1)*y_range],PFR_Height_Energy_LP_width[i*y_range:(i+1)*y_range], color=color_pfr_height_energy_lp, label='PFR Height Energy(LP)',linewidth=3, marker='.', markersize=17, linestyle='-')
        PFR_Height_Energy_LP_Modified = ax.plot(Height_info[i*y_range:(i+1)*y_range],PFR_Height_Energy_LP_Modified_width[i*y_range:(i+1)*y_range], color=color_pfr_height_energy_lp_modified, label='Comp. T-LP Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        # PFR_Height_Energy_LP_1223degC = ax.plot(Height_info[i*y_range:(i+1)*y_range],PFR_Height_Energy_LP_1223degC_width[i*y_range:(i+1)*y_range], color=color_pfr_height_energy_lp_1223_1, label='PFR Height Energy(LP) 1223degC',linewidth=3, marker='.', markersize=17, linestyle='-')
        Comp_W_FR_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],Comp_W_FR_Ctrl_width[i*y_range:(i+1)*y_range], color=color_comp_w_fr_ctrl, label='Comp. T-FR Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        Comp_W_Energy_Ctrl = ax.plot(Height_info[i*y_range:(i+1)*y_range],Comp_W_Energy_Ctrl_width[i*y_range:(i+1)*y_range], color=color_comp_w_energy_ctrl, label='Comp. W-PEF Ctrl',linewidth=3, marker='.', markersize=17, linestyle='-')
        W_FR_Ctrl_PythonWrong = ax.plot(Height_info[i*y_range:(i+1)*y_range],W_FR_Ctrl_PythonWrong_width[i*y_range:(i+1)*y_range], color=color_w_fr_ctrl_python_wrong, label='W-FR Ctrl_PythonCodeWrong-LPNoChange',linewidth=3, marker='.', markersize=17, linestyle='-')
        W_FR_Ctrl_LPMaxOut = ax.plot(Height_info[i*y_range:(i+1)*y_range],W_FR_Ctrl_LPMaxOut_width[i*y_range:(i+1)*y_range], color=color_w_fr_ctrl_lp_max_out, label='W-FR Ctrl_LPMaxOut',linewidth=3, marker='.', markersize=17, linestyle='-')
        W_FR_Ctrl_53Layer = ax.plot(Height_info[i*y_range:(i+1)*y_range],W_FR_Ctrl_53Layer_width[i*y_range:(i+1)*y_range], color=color_w_fr_ctrl_53layer, label='W-FR-Ctrl-53Layer',linewidth=3, marker='.', markersize=17, linestyle='-')

    # plt.ylim(2.4,4.5)

    plt.xlim(0, 55)
    plt.ylim(2.5, 4.2)
    # plt.xticks(np.arange(0, 65, 5))
    # plt.yticks(np.arange(2.7, 4.3, 0.1))
    # Add ticks on the top of the plot and show it with the layer number
    def layer_number(x):
        return (x / 0.8) + 1

    secax = ax.secondary_xaxis('top')
    secax.set_xlabel('Layer #', fontweight="bold", size=30,labelpad=18)
    secax.set_xticks(np.arange(0, 61, 10))
    secax.set_xticklabels([f'{int(layer_number(x))}' for x in np.arange(0, 61, 10)])
    secax.tick_params(axis='both', labelcolor='black', labelsize=30)

    # plt.title('Thin Wall with 1200$^\circ$C Control Method Comparison',fontname="Times New Roman", size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=30)
    plt.xlabel('Height (mm)', fontweight="bold",size=30)
    plt.ylabel('Width (mm)', fontweight="bold",size=30)
    # plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),scatterpoints=1,loc='lower left',ncol=2,fontsize=23.5,markerscale = 1.0)
    # plt.legend([No_Ctrl,PFR_Ctrl,PFR_Height_Ctrl,PFR_Height_Energy_LP],['TW 80 No Control','TW 80 PFR Ctrl','TW 80 PFR Height Ctrl','TW 80 PFR Height Energy(LP)'],scatterpoints=1,loc='lower center',ncol=1,fontsize=17,markerscale = 1.0)

    # Print the arrays after the loop
    print("No_Ctrl_width array:", no_ctrl_array)
    print("PFR_Ctrl_width array:", pfr_ctrl_array)
    print("PFR_Height_Ctrl_width array:", pfr_height_ctrl_array)
    # print("PFR_Height_Energy_LP_width array:", pfr_height_energy_lp_array)
    print("PFR_Height_Energy_LP_Modified_width array:", pfr_height_energy_lp_modified_array)
    # print("PFR_Height_Energy_LP_1223degC_width array:", pfr_height_energy_lp_1223degC_array)
    print("Comp_W_FR_Ctrl_width array:", comp_w_fr_ctrl_array)
    print("Comp_W_Energy_Ctrl_width array:", comp_w_energy_ctrl_array)
    print("W_FR_Ctrl_PythonWrong_width array:", w_fr_ctrl_python_wrong_array)
    print("W_FR_Ctrl_LPMaxOut_width array:", w_fr_ctrl_lp_max_out_array)
    print("W_FR_Ctrl_53Layer_width array:", w_fr_ctrl_53layer_array)

    # Calculate and print the average and standard deviation for each array
    no_ctrl_mean = np.nanmean(no_ctrl_array)
    no_ctrl_std = np.nanstd(no_ctrl_array)
    pfr_ctrl_mean = np.nanmean(pfr_ctrl_array)
    pfr_ctrl_std = np.nanstd(pfr_ctrl_array)
    pfr_height_ctrl_mean = np.nanmean(pfr_height_ctrl_array)
    pfr_height_ctrl_std = np.nanstd(pfr_height_ctrl_array)
    # pfr_height_energy_lp_mean = np.nanmean(pfr_height_energy_lp_array)
    # pfr_height_energy_lp_std = np.nanstd(pfr_height_energy_lp_array)
    pfr_height_energy_lp_modified_mean = np.nanmean(pfr_height_energy_lp_modified_array)
    pfr_height_energy_lp_modified_std = np.nanstd(pfr_height_energy_lp_modified_array)
    # pfr_height_energy_lp_1223degC_mean = np.nanmean(pfr_height_energy_lp_1223degC_array)
    # pfr_height_energy_lp_1223degC_std = np.nanstd(pfr_height_energy_lp_1223degC_array)
    comp_w_fr_ctrl_mean = np.nanmean(comp_w_fr_ctrl_array)
    comp_w_fr_ctrl_std = np.nanstd(comp_w_fr_ctrl_array)
    comp_w_energy_ctrl_mean = np.nanmean(comp_w_energy_ctrl_array)
    comp_w_energy_ctrl_std = np.nanstd(comp_w_energy_ctrl_array)
    w_fr_ctrl_python_wrong_mean = np.nanmean(w_fr_ctrl_python_wrong_array)
    w_fr_ctrl_python_wrong_std = np.nanstd(w_fr_ctrl_python_wrong_array)
    w_fr_ctrl_lp_max_out_mean = np.nanmean(w_fr_ctrl_lp_max_out_array)
    w_fr_ctrl_lp_max_out_std = np.nanstd(w_fr_ctrl_lp_max_out_array)
    w_fr_ctrl_53layer_mean = np.nanmean(w_fr_ctrl_53layer_array)
    w_fr_ctrl_53layer_std = np.nanstd(w_fr_ctrl_53layer_array)

    print(f"No_Ctrl_width mean: {no_ctrl_mean}, std: {no_ctrl_std}")
    print(f"PFR_Ctrl_width mean: {pfr_ctrl_mean}, std: {pfr_ctrl_std}")
    print(f"PFR_Height_Ctrl_width mean: {pfr_height_ctrl_mean}, std: {pfr_height_ctrl_std}")
    # print(f"PFR_Height_Energy_LP_width mean: {pfr_height_energy_lp_mean}, std: {pfr_height_energy_lp_std}")
    print(f"PFR_Height_Energy_LP_Modified_width mean: {pfr_height_energy_lp_modified_mean}, std: {pfr_height_energy_lp_modified_std}")
    # print(f"PFR_Height_Energy_LP_1223degC_width mean: {pfr_height_energy_lp_1223degC_mean}, std: {pfr_height_energy_lp_1223degC_std}")
    print(f"Comp_W_FR_Ctrl_width mean: {comp_w_fr_ctrl_mean}, std: {comp_w_fr_ctrl_std}")
    print(f"Comp_W_Energy_Ctrl_width mean: {comp_w_energy_ctrl_mean}, std: {comp_w_energy_ctrl_std}")
    print(f"W_FR_Ctrl_PythonWrong_width mean: {w_fr_ctrl_python_wrong_mean}, std: {w_fr_ctrl_python_wrong_std}")
    print(f"W_FR_Ctrl_LPMaxOut_width mean: {w_fr_ctrl_lp_max_out_mean}, std: {w_fr_ctrl_lp_max_out_std}")
    print(f"W_FR_Ctrl_53Layer_width mean: {w_fr_ctrl_53layer_mean}, std: {w_fr_ctrl_53layer_std}")
    plt.tight_layout()
    plt.rc('font', weight='bold')
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.73, bottom=0.12)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
    
#*************************Plot section Plot***************************************************   
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(No_Ctrl_df[0]-(No_Ctrl_mid_x-PFR_Ctrl_mid_x), No_Ctrl_df[1],color='black', s=5, marker='.',label='No Ctrl')  
    ax.scatter(PFR_Ctrl_df[0], PFR_Ctrl_df[1],color=color_pfr, s=5, marker='.',label='PFR Ctrl')
    ax.scatter(PFR_Height_Ctrl_df[0]-(PFR_Height_Ctrl_mid_x-PFR_Ctrl_mid_x), PFR_Height_Ctrl_df[1],color=color_pfr_height,s=5 ,  marker='.',label='QPF Ctrl' )
    # ax.scatter(PFR_Height_Energy_LP_df[0]-(PFR_Height_Energy_LP_mid_x-PFR_Ctrl_mid_x), PFR_Height_Energy_LP_df[1],color=color_pfr_height_energy_lp,s=5 ,  marker='.',label='PFR Height Energy(LP)' )
    ax.scatter(PFR_Height_Energy_LP_Modified_df[0]-(PFR_Height_Energy_LP_Modified_mid_x-PFR_Ctrl_mid_x), PFR_Height_Energy_LP_Modified_df[1],color=color_pfr_height_energy_lp_modified,s=5 ,  marker='.',label='Comp. T-LP Ctrl' )
    # ax.scatter(PFR_Height_Energy_LP_1223degC_df[0]-(PFR_Height_Energy_LP_1223degC_mid_x-PFR_Ctrl_mid_x), PFR_Height_Energy_LP_1223degC_df[1],color=color_pfr_height_energy_lp_1223_1,s=5 ,  marker='.',label='PFR Height Energy(LP) 1223degC' )
    ax.scatter(Comp_W_FR_Ctrl_df[0]-(Comp_W_FR_Ctrl_mid_x-PFR_Ctrl_mid_x), Comp_W_FR_Ctrl_df[1],color=color_comp_w_fr_ctrl,s=5 ,  marker='.',label='Comp. T-FR Ctrl' )
    ax.scatter(Comp_W_Energy_Ctrl_df[0]-(Comp_W_Energy_Ctrl_mid_x-PFR_Ctrl_mid_x), Comp_W_Energy_Ctrl_df[1],color=color_comp_w_energy_ctrl,s=5 ,  marker='.',label='Comp. W-PEF Ctrl' )
    ax.scatter(W_FR_Ctrl_PythonWrong_df[0]-(W_FR_Ctrl_PythonWrong_mid_x-PFR_Ctrl_mid_x), W_FR_Ctrl_PythonWrong_df[1],color=color_w_fr_ctrl_python_wrong,s=5 ,  marker='.',label='W-FR Ctrl_PythonCodeWrong-LPNoChange' )
    ax.scatter(W_FR_Ctrl_LPMaxOut_df[0]-(W_FR_Ctrl_LPMaxOut_mid_x-PFR_Ctrl_mid_x), W_FR_Ctrl_LPMaxOut_df[1],color=color_w_fr_ctrl_lp_max_out,s=5 ,  marker='.',label='W-FR Ctrl_LPMaxOut' )
    ax.scatter(W_FR_Ctrl_53Layer_df[0]-(W_FR_Ctrl_53Layer_mid_x-PFR_Ctrl_mid_x), W_FR_Ctrl_53Layer_df[1],color=color_w_fr_ctrl_53layer,s=5 ,  marker='.',label='W-FR-Ctrl-53Layer' )
    plt.legend(scatterpoints=1,loc='lower center',ncol=1,fontsize=23.5,markerscale = 5)
    # plt.title('Thin Wall Side Profile',fontname="Times New Roman", size=17, fontweight="bold")
    ax.tick_params(axis='both', labelcolor='black',labelsize=30)
    plt.xlabel('Width-X coordiante (mm)', fontweight="bold",size=30)
    plt.ylabel('Height-Z coordinate(mm)', fontweight="bold",size=30)
    plt.xlim(7.5, 12.5)
    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.12)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.12)
    plt.rc('font', weight='bold')

#*************************Line in section Plot 1100 and 1200 Comparison***************************************************    
plt.show()
# %%

# %%
