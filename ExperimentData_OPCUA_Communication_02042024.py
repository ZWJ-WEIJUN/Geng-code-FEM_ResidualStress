# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 01:51:50 2024
This code is used to extract the OPC UA communication time data from the 90 layer StepTWs DRY run experiment.
Both StepTWs are with No Energy Ctrl DRY run experiment
@author: Weijun
"""
#V2, include local search function testing

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Rectangle,Patch
from matplotlib.ticker import FormatStrFormatter




# *********************** open a file that recoreding OPC UA read and write data, and save digital image frame for each while loop in python file - START
# python file name: OPCUA_READ_WRITE_TEST.py, located in the BOX folder: https://ucdavis.box.com/s/l3ivxdfxczsm4qakd72cws6x0u6eh8ph
with open('STW70_OPCUA_No_Ctrl_TempFrame_Saved20240204.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_TW_3D_40layers = np.load(f, allow_pickle=True) # Coord_TW_3D_40layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    OPCUA_Read_Time = np.load(f, allow_pickle=True) # OPCUA_Read_Time is the time duration for OPC UA communication read R varaibles from the machine within each frame collected by the IR camera during the experiment
    OPCUA_Write_Time = np.load(f, allow_pickle=True) # OPCUA_Write_Time is the time duration for OPC UA communication write R varibles to the machine within each frame collected by the IR camera during the experiment
    DPSS_angle = np.load(f, allow_pickle=True) # DPSS_angle is the adjustment of the angle on the DPSS for each layer of the step thin wall
    While_loop_time = np.load(f, allow_pickle=True) # While_loop_time is the time duration for the while loop within each frame collected by the IR camera during the experiment and it includes the frame saving time
   


# Print the recorded temperature data collected by the IR camera during the experiment
time_diff = np.diff(While_loop_time)

print(f'The DPSS_angle is: {DPSS_angle}')
print(f"The shape of the Whileloop_time array is {While_loop_time.shape}")
print(f'The While_loop_time is: {While_loop_time}')
# Print the time difference
print(f"The shape of the Whileloop_time array is {While_loop_time.shape}")
print(f"The time difference between each loop is: {time_diff}")
print(f"The shape of the time difference is: {time_diff.shape}")
print(f'Frame collected (np.ndarray): {Frame_history.shape}') # Because in the Python data collection code, the first frame is defined as 'np.zeros([palette_width.value * palette_height.value], dtype=np.unit8)', so the time stamp is not recorded for the first frame, so the shape of the Time_stamp is 1 less than the shape of the Frame_history. 
# Due to the time stamp is not recorded for the last frame, so the time to caputre for the last frame is unknown.
print(f'Frame index: {Frame_index}')
print(f'Frame index shape: {Frame_index.shape}')
# Calculate the median temperature for each layer
Medium_temperature_OriginalRectSearch = np.median(MaxT_OriginRectSearch_alllayers, axis=1)   # Median temperature for each layer based on the rectangular (31column x 7row) search method
# When axis = 1, it means that the medians will be computed along each row of the input array. In other words, the function will calculate the median value for each row separately.
print(f"The median temperature for each layer based on the rectangular (31column x 7row) search method is: {Medium_temperature_OriginalRectSearch}")

print(f"The OPCUA_Read_Time is: {OPCUA_Read_Time}")
print(f"The shape of the OPCUA_Read_Time is: {OPCUA_Read_Time.shape}")
OPCUA_Read_Time = OPCUA_Read_Time[:-1] # I would like to igonore the last element of the OPCUA_Read_time because the While_loop_time does not have data to calculate the last loop time duration, I needs to sync OPCUA_Read_Time and While_loop_time to be on the same plot
print(f"The OPCUA_Write_Time is: {OPCUA_Write_Time}")
print(f"The shape of the OPCUA_Write_Time is: {OPCUA_Write_Time.shape}")

# Plot the time difference
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# Generate an array of indices
indices = np.arange(Frame_history.shape[0] -2 ) # the shape of the Frame_history is (2150, 288, 382), so the first number in its shpae array is 2150.
# Here -2 means minus the first frame with all zero pixels and minus the last frame which does not have time recorded
plt.scatter(indices, time_diff, color='tab:blue', label='Each while loop time duration', s=7)
plt.scatter(indices, OPCUA_Read_Time, color='orange', label='OPCUA read time duration', s=7)
plt.scatter(Frame_index, OPCUA_Write_Time, color='black', label='OPCUA write time duration', s=7)
plt.xlabel('While Loop Index')
plt.ylabel('Time Spent (s)')
plt.title('Python Recording Duration for ALL Frames Captured with an IR Camera and Saved as .npy Files') # Each while loop including: OPC UA communication read R varaibles from the machine, OPC UA communication write R varibles to the machine, DPSS angle adjustment, and frame reading and saving time


legend = plt.legend() 
# Get the legend's lines and texts
texts = legend.get_texts()
# Set the color of each text
colors = ['tab:blue', 'orange', 'black']
for color, text in zip(colors, texts):
    text.set_color(color)

# Set the y-axis ticks
plt.yticks(np.arange(0.0, 1.4, 0.1))
# *********************** open a file that recoreding OPC UA read and write data, and save digital image frame for each while loop in python file - END













# *********************** open a file that recoreding OPC UA read and write data, and NO save digital image frame for each while loop in python file - START
with open('STW70_OPCUA_No_Ctrl20240204.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_TW_3D_40layers = np.load(f, allow_pickle=True) # Coord_TW_3D_40layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    OPCUA_Read_Time = np.load(f, allow_pickle=True) # OPCUA_Read_Time is the time duration for OPC UA communication read R varaibles from the machine within each frame collected by the IR camera during the experiment
    OPCUA_Write_Time = np.load(f, allow_pickle=True) # OPCUA_Write_Time is the time duration for OPC UA communication write R varibles to the machine within each frame collected by the IR camera during the experiment
    DPSS_angle = np.load(f, allow_pickle=True) # DPSS_angle is the adjustment of the angle on the DPSS for each layer of the step thin wall
    While_loop_time = np.load(f, allow_pickle=True) # While_loop_time is the time duration for the while loop within each frame collected by the IR camera during the experiment and it includes the frame saving time
   


# Print the recorded temperature data collected by the IR camera during the experiment
time_diff = np.diff(While_loop_time)

print(f'The DPSS_angle is: {DPSS_angle}')
print(f"The shape of the Whileloop_time array is {While_loop_time.shape}")
print(f'The While_loop_time is: {While_loop_time}')
# Print the time difference
print(f"The time difference between each loop is: {time_diff}")
print(f"The shape of the time difference is: {time_diff.shape}")
print(f'Frame collected (np.ndarray): {Frame_history.shape}') # Because in the Python data collection code, the first frame is defined as 'np.zeros([palette_width.value * palette_height.value], dtype=np.unit8)', so the time stamp is not recorded for the first frame, so the shape of the Time_stamp is 1 less than the shape of the Frame_history. 
# Due to the time stamp is not recorded for the last frame, so the time to caputre for the last frame is unknown.
print(f'Frame index: {Frame_index}')
print(f'Frame index shape: {Frame_index.shape}')
# Calculate the median temperature for each layer
Medium_temperature_OriginalRectSearch = np.median(MaxT_OriginRectSearch_alllayers, axis=1)   # Median temperature for each layer based on the rectangular (31column x 7row) search method
# When axis = 1, it means that the medians will be computed along each row of the input array. In other words, the function will calculate the median value for each row separately.
print(f"The median temperature for each layer based on the rectangular (31column x 7row) search method is: {Medium_temperature_OriginalRectSearch}")

print(f"The OPCUA_Read_Time is: {OPCUA_Read_Time}")
print(f"The shape of the OPCUA_Read_Time is: {OPCUA_Read_Time.shape}")
OPCUA_Read_Time = OPCUA_Read_Time[:-1] # I would like to igonore the last element of the OPCUA_Read_time because the While_loop_time does not have data to calculate the last loop time duration, I needs to sync OPCUA_Read_Time and While_loop_time to be on the same plot
print(f"The OPCUA_Write_Time is: {OPCUA_Write_Time}")
print(f"The shape of the OPCUA_Write_Time is: {OPCUA_Write_Time.shape}")



# Plot the time difference
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# Generate an array of indices
indices = np.arange(While_loop_time.shape[0] - 1 ) # the shape of the While_loop_time is (2149,), so the first number in its shpae array is 2149.
# Here -2 means minus the first frame with all zero pixels and minus the last frame which does not have time recorded
plt.scatter(indices, time_diff, color='tab:blue', label='Each while loop time duration', s=7)
plt.scatter(indices, OPCUA_Read_Time, color='orange', label='OPCUA read time duration', s=7)
plt.scatter(Frame_index, OPCUA_Write_Time, color='black', label='OPCUA write time duration', s=7)
plt.xlabel('While Loop Index')
plt.ylabel('Time Spent (s)')
plt.title('Python Recording Duration for NO Frames Captured with an IR Camera or Saved as .npy Files') # Each while loop including: OPC UA communication read R varaibles from the machine, OPC UA communication write R varibles to the machine, DPSS angle adjustment, and frame reading and saving time


legend = plt.legend() 
# Get the legend's lines and texts
texts = legend.get_texts()
# Set the color of each text
colors = ['tab:blue', 'orange', 'black']
for color, text in zip(colors, texts):
    text.set_color(color)

# Set the y-axis ticks
plt.yticks(np.arange(0.0, 1.4, 0.1))
# *********************** open a file that recoreding OPC UA read and write data, and NO save digital image frame for each while loop in python file - END
plt.show()

