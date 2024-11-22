# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:42:50 2024
This code is used to extract the temperature data from the experiment IR data
V2: Recrtangular search testing and data extraction of original data from experiment 
V3: Shifting method base on maximum tempearture and data extraction of original data from experiment

@author: muqin/Weijun
"""
#V2, include local search function testing

# import matplotlib
# matplotlib.use('Agg')  # This will set the backend to 'Agg', which is a non-interactive backend. This way, the plots will not be displayed during the debug execution of the code, and you will be able to step over the code without any issues in the debug processss.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Rectangle,Patch
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import griddata



# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - START

def get_image_coord_ls(Layer_num: int,Frame_index,Frame_history):
    """
    Description: This function is used to get the image coordinate of the thin wall geometry in the 3D printing space

    Input: 
    Layer_num: the layer number of the thin wall, from 1 to 40, no Zero
    Frame_index: the index of the frame that collected by the IR camera by the laser head moves away from the thin wall in 115mm distance (See G-code for more details https://ucdavis.box.com/s/67t2ogax3nkozbp8f9tc4a4sy6bcqimw)
    Frame_history: the temperature data collected by the IR camera during the experiment frame by frame

    Output: 
    image_coord_ls: the image coordinate of the thin wall geometry in the 3D printing space
    MaxT_RecSearch_perlayer: the max temperature acquired based on new local search method -  rectangular search method - for 8 points in each layer
    """
    x_start = 90.
    y_start = 8.
    z_start = -4.               # Here use -4 is due to the 4mm thickness of the calibration black gasket plate on the top of the Z0 substrate, otherwise the z_start should be 0.

    Layer_Height = 0.7
    Length = 70.                # length of the thin wall in the y direction

    L_Change = 20                 # change of the length for step thin wall in the y direction
    Layer_Change = 30            # number of layers
    N_Change = 3                 # number of changes of the length for step thin wall in the y direction
    N_Data_perLayer = 7          # number of data points collected in each layer
    Total_Layer = Layer_Change*N_Change  # total number of layers calculation for the step thin wall

    Coord_TW_3D = []

    x_coord = x_start 
    y_coord = y_start 
    z_coord = z_start+Layer_Height   
    """??? 1. Don't understand why -4 + 0.7 = -3.3 ???"""



    # For look here is to get the coordinate of the thin wall geometry in the 3D printing space
    # Total 630 data points, 7 data points per layer, 90 layers
    if not Coord_TW_3D:
        for i in range(N_Change):
            x_i = x_start
            y_i = y_start + 5 + i*L_Change/2            # y_i is the initial point for the step thin wall; Here +5 is to make sure the first location for collecting data is not on the edge of the thin wall but at y = 13 mm
            y_f = y_start - 5 + Length - i*L_Change/2   # y-f is the final point for the step thin wall; Here -5 is to make sure the last location for collecting data is not on the edge of the thin wall but at y = 73 mm   
            z_i = z_start     
            L_inc = (y_f-y_i)/(N_Data_perLayer - 1)   # L_inc is the increment of the length for step thin wall in the y direction, in this case is 73 - 13 / 7 ~= 8.57
            for j in range(Layer_Change):
                x_coord = x_i
                y_coord = y_i
                Coord_TW_3D.append([x_coord,y_coord,z_coord])
                for k in range(N_Data_perLayer - 1):
                    # update the y coordinate in each layer
                    y_coord = y_coord + L_inc
                    y_coord  = round(y_coord,3)
                    Coord_TW_3D.append([x_coord,y_coord,z_coord])
                # update the z coordinate after each layer
                z_coord = round(z_coord+Layer_Height,3)
        # Coord_TW_3D - a list contains 630 data points, 7 data points per layer, 90 layers

    # Extract Image Coordinate
    Indexi = (Layer_num - 1) *(N_Data_perLayer - 1)+(Layer_num-1)   # Indexi is the start index of the data points for the current layer
    Indexf= (Layer_num)*(N_Data_perLayer-1)+Layer_num # Indexf is the end index of the data points for the current layer
    Coor_CurrLayer = Coord_TW_3D[Indexi:Indexf]                       # Coor_CurrLayer is the list of the 8 eight coordinate for the current layer
                                                            # The shape of Coord_TW_3D is: (320, 3)
    Coor_CurrLayer = np.array(Coor_CurrLayer)                             # Convert Coor_CurrLayer to a numpy array
    Length_CurrLayer = Coor_CurrLayer[N_Data_perLayer-1, 1] - Coor_CurrLayer[0, 1]       # Length_CurrLayer is the length between end temperature collecting point and the start temperature collecting point in the current layer.
    print(f"The distance between start collection pt and end collection pt of one layer thin wall is: {Length_CurrLayer} mm")



    date_TranMat = '20240121'
    TranMat = np.load('TranMat_'+date_TranMat+'.npy')    # Size 2 x 4, 2 rows and 4 columns
    # print(f"{TranMat.shape} is the shape of the TranMat matrix.")

    Coor_CurrLayer_trans = np.append(Coor_CurrLayer, np.array([1]*(len(Coor_CurrLayer))).reshape(-1,1), axis=1) # Add a column of 1 to the Coor_CurrLayer array
    """??? 2. why add a column of 1 to the Coor_CurrLayer array???"""
    # [[1]
    # [1]
    # [1]
    # [1]
    # [1]
    # [1]
    # [1]
    # [1]] is added to the end of each row in Coor_CurrLayer, so that the transformation matrix can be applied to the Coor_CurrLayer array
    # For example, the first layer of Coor_CurrLayer is: 
    #   [[20.    13.    -3.3  ]
    #    [20.    21.571 -3.3  ]
    #    [20.    30.142 -3.3  ]
    #    [20.    38.713 -3.3  ]
    #    [20.    47.284 -3.3  ]
    #    [20.    55.855 -3.3  ]
    #    [20.    64.426 -3.3  ]
    #    [20.    72.997 -3.3  ]]
    # After np.append(Coor_CurrLayer, np.array([1]*(len(Coor_CurrLayer))).reshape(-1,1), axis=1), now the Coor_CurrLayer is:
    #  [[20.    13.    -3.3    1.   ]
    #   [20.    21.571 -3.3    1.   ]
    #   [20.    30.142 -3.3    1.   ]
    #   [20.    38.713 -3.3    1.   ]
    #   [20.    47.284 -3.3    1.   ]
    #   [20.    55.855 -3.3    1.   ]
    #   [20.    64.426 -3.3    1.   ]
    #   [20.    72.997 -3.3    1.   ]]


    image_coord_ls = [] 
    MaxT_RecSearch_perlayer = []                                           # max temperature from rectangular local search
    ic_MT_AFTER_RecSearch = []                                   # the search coord of the max temperature from rectangular local search


    # Define the size of the rectangualr search area
    column = 15
    row = 2

  
    Frame_index_per_layer = Frame_index[Layer_num]
    temperature_data = Frame_history[Frame_index_per_layer+1] # In the original Geng's fuzzy ctrl code, the first FOUR element of Frame_index is [0, 0, 1, 2...], the first one is repeated, so here need to add 1 to frame_index_i
    # temperature_data is the temperature data for the each layer, shape is (288, 382) - Please note that 288 rows (y) and 382 columns (x)

    # For loop here is to get the 2D image coordinate of the thin wall geometry in the 3D printing space for each point in each layer
    for mc in Coor_CurrLayer_trans:
        ic = np.round(np.matmul(TranMat,mc)).astype(int)   # mc (4x1, i.e., [20.    13.    -3.3    1.   ] ) means each machine coordinate in each layer, which is the coordinate of the thin wall geometry in the 3D printing space, after the transformation, it becomes the 2D image coordinate (ic) (2x1).
        # The astype() method creates a copy of the array, and converts the type of the data in the copy. In this case, it's converting the data to int, which means integer

        image_coord_ls.append(ic)  # Append the 2D image coordinate (x, y) of the thin wall geometry in the 3D printing space to the image_coord_ls list
        # print(ic)             # Print the 2D image coordinate (x, y) or (column, row) of the thin wall geometry in the 3D printing space. e.g. the first point of ic is array([103 219]) in the image frame of the first layer
        """??? 3. how the variable ic is defined???ic[0] is the x coordinate, ic[1] is the y coordinate (row), the origin is the top left corner of the image frame, the x axis is the horizontal axis(column), the y axis is the vertical axis (row), the unit is pixel ???"""
        
        local_search_Rec = temperature_data[ic[1]-row:ic[1]+row+1,ic[0]-column:ic[0]+column+1]              # local search rectangular area of 21x7 pixels (21columns x 7rows)
        if np.any(local_search_Rec):
            local_high_temp = np.max(local_search_Rec)                                                      # the max temperature in the local search area
            index_local_search_Rec = np.unravel_index(np.argmax(local_search_Rec), np.shape(local_search_Rec))                   # the index of the max temperature in the local search area in the form of tuple with two elements (x,y) or (column, row)
            max_temp_coord = [index_local_search_Rec[1]+ic[0]-column,index_local_search_Rec[0]+ic[1]-row]                         # max_temp_coord is the coordinate of the max temperature in the big image frame after locaL serach, not in the local search area
        else:
            local_high_temp = temperature_data[ic[1],ic[0]]      #If the local search area is empty, then the max temperature is the temperature at the current point in the big image frame
            """Here is confusing, the temperature_data is 288rows by 382columns, if the origin is the top left corner of the image frame, the x axis is the horizontal axis(column), the y axis is the vertical axis (row),
            so the temepratre data is y by x, but the varaible ic is x (column) by y(row), so the temperature_data[ic[1],ic[0]] is the correct way to get the temperature at the current point in the big image frame"""
        MaxT_RecSearch_perlayer.append(local_high_temp)
        ic_MT_AFTER_RecSearch.append(max_temp_coord)    # Append the coordinate of the max temperature in the local search area to the ic_MT_AFTER_RecSearch lis

   
    image_coord_ls = np.array(image_coord_ls)          # Convert the image_coord_ls list to a numpy array
    MaxT_RecSearch_perlayer = np.array(MaxT_RecSearch_perlayer)            # Max temperature acquired based on rectangular local search method for 8 points in each layer
    ic_MT_AFTER_RecSearch = np.array(ic_MT_AFTER_RecSearch)           # The coordinate of the max temperature in the local rectangular search area for 8 points in each layer
    
    
    # return image_coord_ls, MaxT_RecSearch_perlayer, ic_MT_AFTER_RecSearch, ic_MT_AFTER_SquareSearch, y_i, y_f, N_Data_perLayer
    return image_coord_ls, MaxT_RecSearch_perlayer, ic_MT_AFTER_RecSearch, y_i, y_f, N_Data_perLayer, Length_CurrLayer


# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - END



with open('STW70_NO_CTRL_run20240131.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_TW_3D_40layers = np.load(f, allow_pickle=True) # Coord_TW_3D_40layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    Time_Stamp = np.load(f, allow_pickle=True) # Time_Stamp is the time stamp for each frame collected by the IR camera during the experiment


# Print the recorded temperature data collected by the IR camera during the experiment
time_diff = np.diff(Time_Stamp)
# Print the time difference
print(f"The shape of the Time_stamp is {Time_Stamp.shape}")
print(f"The time difference between each loop is: {time_diff}")
print(f"The shape of the time difference is: {time_diff.shape}")
print(f'Frame collected (np.ndarray): {Frame_history.shape}') # Because in the Python data collection code, the first frame is defined as 'np.zeros([palette_width.value * palette_height.value], dtype=np.unit8)', so the time stamp is not recorded for the first frame, so the shape of the Time_stamp is 1 less than the shape of the Frame_history. 
# Due to the time stamp is not recorded for the last frame, so the time to caputre for the last frame is unknown.
print(f'Frame index: {Frame_index}')
# Calculate the median temperature for each layer
Medium_temperature_OriginalRectSearch = np.median(MaxT_OriginRectSearch_alllayers, axis=1)   # Median temperature for each layer based on the rectangular (31column x 7row) search method
# When axis = 1, it means that the medians will be computed along each row of the input array. In other words, the function will calculate the median value for each row separately.
print(f"The median temperature for each layer based on the rectangular (31column x 5row) search method is: {Medium_temperature_OriginalRectSearch}")

# Save the Medium_temperature_OriginalRectSearch array to a file - Only needs one time
# np.save('SPTW_NOCtrl_01312024_Medium_temperature_RecSearch_5(row)x31(column).npy', Medium_temperature_OriginalRectSearch)


# Create new temperature distribution with 31(column) by 5(row)rectangualr search methond
MaxT_RecSearch_alllayers =[]
for layer_num in range (1,len(frame_index)):
   imge_coord_perLayer, MaxT_RecSearch_perlayer,ic_MT_AFTER_RecSearch, y_i, y_f, N_Data_perLayer, Length_CurrLayer_mm = get_image_coord_ls(layer_num,frame_index,Frame_history)
   MaxT_RecSearch_alllayers.append(MaxT_RecSearch_perlayer)
print(MaxT_OriginRectSearch_alllayers.shape)

Medium_temperature_RecSearch = np.median(MaxT_RecSearch_alllayers, axis=1)   # Median temperature for each layer based on the new local search method - rectangular search method


plt.rcParams["font.weight"] = "bold"
layer = np.arange(1, MaxT_OriginRectSearch_alllayers.shape[0]+1, dtype=int)
# Extract the first sublist from Coord_TW_3D_40layers
first_sublist = Coord_TW_3D_40layers[0]
# Extract the second element from each sublist in first_sublist
second_elements = [sublist[1] for sublist in first_sublist]

x_labels = np.array(second_elements, dtype=int)
MaxT_OriginRectSearch_alllayers = np.array(MaxT_OriginRectSearch_alllayers, dtype=float)




# Plot the time difference
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# Generate an array of indices
indices = np.arange(Frame_history.shape[0] - 2) # the shape of the Frame_history is (1518, 288, 382), so the first number in its shpae array is 1518.
# Here -2 means minus the first frame with all zero pixels and minus the last frame which does not have time recorded
plt.scatter(indices, time_diff)
plt.xlabel('Frame Index')
plt.ylabel('Time Spent (s)')
plt.title('Time Spent For Each Frame Collected by the IR Camera', fontname="Arial Black",
          size=15, fontweight="bold")



# ***********************  Heatmap plot - START
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
plt.imshow(MaxT_OriginRectSearch_alllayers, cmap='coolwarm', interpolation='nearest',  origin='lower', aspect=0.4)  # Create a heatmap with the data
cbar =plt.colorbar(label='Temperature (°C)', shrink=0.8)  # Add a colorbar to the right of the plot
# cbar.ax.margins(y=0)  # Make the colorbar narrower

# Add numbers to each cell
for i in range(MaxT_OriginRectSearch_alllayers.shape[0]):
    for j in range(MaxT_OriginRectSearch_alllayers.shape[1]):
        plt.text(j, i, format(MaxT_OriginRectSearch_alllayers[i, j], '.1f'),
                 horizontalalignment="center",
                 verticalalignment = "center",
                 color="white" if MaxT_OriginRectSearch_alllayers[i, j] > 900 else "black", fontsize=8)

# Set up the labels for the x and y axes
plt.xlabel('Coordinate along machine Y axis (mm)')
plt.ylabel('Layers #')

# Set up the ticks for the y axis
plt.yticks(ticks=np.arange(MaxT_OriginRectSearch_alllayers.shape[0]), labels=np.arange(1, MaxT_OriginRectSearch_alllayers.shape[0] + 1)) # MaxT_OriginRectSearch_alllayers.shape[0] = 40
# Set up the ticks for the x axis
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)

plt.title('Temperature Distribution - origianl Rectangular Search 31 (column) x 5 (row) pixels - Experimental Data')  # Add a title to the plot
#***********************  Heatmap plot - END


# MaxT_RecSearch_alllayers = np.array(MaxT_RecSearch_alllayers, dtype=float)

# # ***********************  Heatmap plot_Rectangular Search new figure with a custom size
# plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# plt.imshow(MaxT_RecSearch_alllayers, cmap='coolwarm', interpolation='nearest',  origin='lower', aspect=0.4)  # Create a heatmap with the data
# cbar =plt.colorbar(label='Temperature (°C)', shrink=0.8)  # Add a colorbar to the right of the plot
# # cbar.ax.margins(y=0)  # Make the colorbar narrower

# # Add numbers to each cell
# for i in range(MaxT_RecSearch_alllayers.shape[0]):
#     for j in range(MaxT_RecSearch_alllayers.shape[1]):
#         plt.text(j, i, format(MaxT_RecSearch_alllayers[i, j], '.1f'),
#                  horizontalalignment="center",
#                  verticalalignment = "center",
#                  color="white" if MaxT_OriginRectSearch_alllayers[i, j] > 900 else "black", fontsize=8)

# # Set up the labels for the x and y axes
# plt.xlabel('Coordinate along machine Y axis (mm)')
# plt.ylabel('Layers #')

# # Set up the ticks for the y axis
# plt.yticks(ticks=np.arange(MaxT_OriginRectSearch_alllayers.shape[0]), labels=np.arange(1, MaxT_OriginRectSearch_alllayers.shape[0] + 1)) # MaxT_OriginRectSearch_alllayers.shape[0] = 40
# # Set up the ticks for the x axis
# plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)

# plt.title('Temperature Distribution - Rectangular Search 31 (column) x 5 (row) pixels - Verification based on the captured temeprature frame')  # Add a title to the plot
# #***********************  Heatmap plot - END
# plt.show()




# ***********************  3D Surface plot No_Ctrl & LP_Ctrl - START

fig = plt.figure(figsize=(15, 40))  # Increase the figure size to make it taller
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

# Create the x, y, and z coordinates for the surface plot
x = np.arange(MaxT_OriginRectSearch_alllayers.shape[1])  # column index         
y = np.arange(1, MaxT_OriginRectSearch_alllayers.shape[0] + 1)  # row (Thin Wall Layer #) index starting from 1
X, Y = np.meshgrid(x, y)
Z = MaxT_OriginRectSearch_alllayers
print(MaxT_OriginRectSearch_alllayers)

# Use interpolation to make the surface plot smoother
X_smooth = np.linspace(X.min(), X.max(), 200)
Y_smooth = np.linspace(Y.min(), Y.max(), 1000)
X_smooth, Y_smooth = np.meshgrid(X_smooth, Y_smooth)
Z_smooth = griddata((X.flatten(), Y.flatten()), Z.flatten(), (X_smooth, Y_smooth), method='cubic')

# Create the surface plot with the smoothed data
surf = ax.plot_surface(X_smooth, Y_smooth, Z_smooth, cmap='Greys', edgecolor='none', alpha=0.5)  # Remove the mesh


# # Set up the ticks for the x and y axes
# ax.set_xticks(ticks=np.arange(len(x_labels)), labels=x_labels)
# ax.set_yticks(y)


# ***********************  3D Surface plot add LP_Ctrl - START
with open('STW70_LP_CTRL_run20240204.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers_LPCtrl = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers_LPCtrl is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_STW_3D_90layers_LPCtrl = np.load(f, allow_pickle=True) # Coord_STW_3D_90layers_LPCtrl is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    # lp_array_LPCtrl = np.load(f, allow_pickle=True)
    # fr_array_LPCtrl = np.load(f, allow_pickle=True)
    # Frame_history_LPCtrl = np.load(f, allow_pickle=True) # Frame_history_LPCtrl is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    # frame_index_LPCtrl = np.load(f, allow_pickle=True)
    # # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    # Frame_index_LPCtrl = frame_index_LPCtrl[1:] # Frame_index_LPCtrl is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall
    # OPCUA_Read_Time_LPCtrl = np.load(f, allow_pickle=True) # OPCUA_Read_Time_LPCtrl is the time duration for OPC UA communication read R varaibles from the machine within each frame collected by the IR camera during the experiment
    # OPCUA_Write_Time_LPCtrl = np.load(f, allow_pickle=True) # OPCUA_Write_Time_LPCtrl is the time duration for OPC UA communication write R varibles to the machine within each frame collected by the IR camera during the experiment


MaxT_OriginRectSearch_alllayers_LPCtrl = np.array(MaxT_OriginRectSearch_alllayers_LPCtrl, dtype=float)
# Extract the first sublist from Coord_STW_3D_90layers_LPCtrl
print(f"The coord_STW_3D_90layers_LPCtrl is: {Coord_STW_3D_90layers_LPCtrl}")


# Extract the first sublist from Coord_STW_3D_90layers_LPCtrl
first_sublist_LPCtrl = Coord_STW_3D_90layers_LPCtrl[0]
# Extract the second element from each sublist in first_sublist
second_elements_LPCtrl_1 = [sublist[1] for sublist in first_sublist_LPCtrl]
x_labels_LPCtrl_1 = np.round(np.array(second_elements_LPCtrl_1, dtype=float), 1)


# *********************************** Code saved for plot temperature distribution for three differnt thin wall length geometry - START ***********************************
# # Extract the sublist at index 29 from Coord_STW_3D_90layers_LPCtrl
# first_sublist_LPCtrl_2 = Coord_STW_3D_90layers_LPCtrl[30]
# # Extract the second element from each sublist in first_sublist_LPCtrl_2
# second_elements_LPCtrl_2 = [sublist[1] for sublist in first_sublist_LPCtrl_2]
# x_labels_LPCtrl_2 = np.round(np.array(second_elements_LPCtrl_2, dtype=float), 1)

# # Extract the sublist at index 59 from Coord_STW_3D_90layers_LPCtrl
# first_sublist_LPCtrl_3 = Coord_STW_3D_90layers_LPCtrl[60]
# # Extract the second element from each sublist in first_sublist_LPCtrl_3
# second_elements_LPCtrl_3 = [sublist[1] for sublist in first_sublist_LPCtrl_3]
# x_labels_LPCtrl_3 = np.round(np.array(second_elements_LPCtrl_3, dtype=float), 1)



# # Create the x, y, and z coordinates for the surface plot
# x_LPCtrl = np.arange(MaxT_OriginRectSearch_alllayers_LPCtrl.shape[1])  # column index
# y_LPCtrl = np.arange(1, MaxT_OriginRectSearch_alllayers_LPCtrl.shape[0] + 1)  # row (Thin Wall Layer #) index starting from 1
# X_LPCtrl, Y2_LPCtrl = np.meshgrid(x_LPCtrl, y_LPCtrl)
# Z_LPCtrl = MaxT_OriginRectSearch_alllayers_LPCtrl
# # print(MaxT_OriginRectSearch_alllayers_LPCtrl)

# # Use interpolation to make the surface plot smoother
# X_smooth_LPCtrl = np.linspace(X_LPCtrl.min(), X_LPCtrl.max(), 200)
# Y_smooth_LPCtrl = np.linspace(Y2_LPCtrl.min(), Y2_LPCtrl.max(), 1000)
# X_smooth_LPCtrl, Y_smooth_LPCtrl = np.meshgrid(X_smooth_LPCtrl, Y_smooth_LPCtrl)
# Z_smooth_LPCtrl = griddata((X_LPCtrl.flatten(), Y2_LPCtrl.flatten()), Z_LPCtrl.flatten(), (X_smooth_LPCtrl, Y_smooth_LPCtrl), method='cubic')

# # Create the surface plot with the smoothed data
# surf2 = ax.plot_surface(X_smooth_LPCtrl, Y_smooth_LPCtrl, Z_smooth_LPCtrl, cmap='Reds', edgecolor='none')  # Remove the mesh

# # Modify the x-axis ticks for the surface plot
# ax.set_xticks(ticks=np.arange(len(x_labels_LPCtrl_1[:8])), labels=x_labels_LPCtrl_1[:8])  # For the 1st row to 30th row
# ax.set_xticks(ticks=np.arange(len(x_labels_LPCtrl_2[:8])), labels=x_labels_LPCtrl_2[:8])  # For the 31st row to 60th row
# ax.set_xticks(ticks=np.arange(len(x_labels_LPCtrl_3[:8])), labels=x_labels_LPCtrl_3[:8])  # For the 61st row to 90th row   
# *********************************** Code saved for plot temperature distribution for three differnt thin wall length geometry - END ***********************************


# Create the x, y, and z coordinates for the surface plot
x_LPCtrl = np.arange(MaxT_OriginRectSearch_alllayers_LPCtrl.shape[1])  # column index
y_LPCtrl = np.arange(1, MaxT_OriginRectSearch_alllayers_LPCtrl.shape[0] + 1)  # row (Thin Wall Layer #) index starting from 1
X_LPCtrl, Y_LPCtrl = np.meshgrid(x_LPCtrl, y_LPCtrl)
Z_LPCtrl = MaxT_OriginRectSearch_alllayers_LPCtrl
# print(MaxT_OriginRectSearch_alllayers_LPCtrl)

# Use interpolation to make the surface plot smoother
X_smooth_LPCtrl = np.linspace(X_LPCtrl.min(), X_LPCtrl.max(), 200)
Y_smooth_LPCtrl = np.linspace(Y_LPCtrl.min(), Y_LPCtrl.max(), 1000)
X_smooth_LPCtrl, Y_smooth_LPCtrl = np.meshgrid(X_smooth_LPCtrl, Y_smooth_LPCtrl)
Z_smooth_LPCtrl = griddata((X_LPCtrl.flatten(), Y_LPCtrl.flatten()), Z_LPCtrl.flatten(), (X_smooth_LPCtrl, Y_smooth_LPCtrl), method='cubic')

# Create the surface plot with the smoothed data
surf_LPCtrl = ax.plot_surface(X_smooth_LPCtrl, Y_smooth_LPCtrl, Z_smooth_LPCtrl, cmap='Reds', edgecolor='none')  # Remove the mesh

# Set up the ticks for the x and y axes
ax.set_xticks(ticks=np.arange(len(x_labels_LPCtrl_1)), labels=x_labels_LPCtrl_1)
# Set up the ticks for the y axis with larger spacing
ax.set_yticks(ticks=np.arange(1, MaxT_OriginRectSearch_alllayers.shape[0] + 1, 10))
# ***********************  3D Surface plot add LP_Ctrl - END

# ***********************  3D Surface plot add FR_Ctrl - START
with open('STW70_Fr_Ctrl_run20240204.npy', 'rb') as f:
    MaxT_OriginRectSearch_alllayers_FRCtrl = np.load(f, allow_pickle=True)  # MaxT_OriginRectSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_STW_3D_90layers_FRCtrl = np.load(f, allow_pickle=True) # Coord_STW_3D_90layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  

MaxT_OriginRectSearch_alllayers_FRCtrl = np.array(MaxT_OriginRectSearch_alllayers_FRCtrl, dtype=float)
# Extract the first sublist from Coord_TW_3D_40layers_FRCtrl
print(f"The coord_STW_3D_90layers_FRCtrl is: {Coord_STW_3D_90layers_FRCtrl}")

# Extract the first sublist from Coord_TW_3D_40layers_FRCtrl
first_sublist_FRCtrl = Coord_STW_3D_90layers_FRCtrl[0]
# Extract the second element from each sublist in first_sublist
second_elements_FRCtrl_1 = [sublist[1] for sublist in first_sublist_FRCtrl]
x_labels_FRCtrl_1 = np.array(second_elements_FRCtrl_1, dtype=int)

# Create the x, y, and z coordinates for the surface plot
x_FRCtrl = np.arange(MaxT_OriginRectSearch_alllayers_FRCtrl.shape[1])  # column index
y_FRCtrl = np.arange(1, MaxT_OriginRectSearch_alllayers_FRCtrl.shape[0] + 1)  # row (Thin Wall Layer #) index starting from 1
X_FRCtrl, Y_FRCtrl = np.meshgrid(x_FRCtrl, y_FRCtrl)
Z_FRCtrl = MaxT_OriginRectSearch_alllayers_FRCtrl
# print(MaxT_OriginRectSearch_alllayers_FRCtrl)

# Use interpolation to make the surface plot smoother
X_smooth_FRCtrl = np.linspace(X_FRCtrl.min(), X_FRCtrl.max(), 200)
Y_smooth_FRCtrl = np.linspace(Y_FRCtrl.min(), Y_FRCtrl.max(), 1000)
X_smooth_FRCtrl, Y_smooth_FRCtrl = np.meshgrid(X_smooth_FRCtrl, Y_smooth_FRCtrl)
Z_smooth_FRCtrl = griddata((X_FRCtrl.flatten(), Y_FRCtrl.flatten()), Z_FRCtrl.flatten(), (X_smooth_FRCtrl, Y_smooth_FRCtrl), method='cubic')

# Create the surface plot with the smoothed data
surf_FRCtrl = ax.plot_surface(X_smooth_FRCtrl, Y_smooth_FRCtrl, Z_smooth_FRCtrl, cmap='Purples', edgecolor='none')  # Remove the mesh

# Set up the ticks for the x and y axes
ax.set_xticks(ticks=np.arange(len(x_labels_FRCtrl_1)), labels=x_labels_FRCtrl_1)
ax.set_yticks(ticks=np.arange(0, MaxT_OriginRectSearch_alllayers.shape[0]+10, 10))

# Adjust the spacing between the ticks and labels on the x-axis
ax.tick_params(axis='x', which='major', pad=2)
# Adjust the spacing between the ticks and labels on the y-axis
ax.tick_params(axis='y', which='major', pad=2)
ax.tick_params(axis='z', which='major', pad=2)

# Enlarge the font size and ticks size
ax.tick_params(axis='both', which='major', labelsize=16)
ax.set_xlabel('Coordinate along machine Y axis (mm)', fontname='Arial Black', fontsize=20)
ax.set_ylabel('Layer #', fontname='Arial Black', fontsize=20)
ax.set_zlabel('Temperature (°C)', fontname='Arial Black', fontsize=20)

# Stretch the plot along the z and y axes
ax.dist = 12
ax.set_box_aspect([4, 8, 8])  # Adjust the aspect ratio of the plot

# Set up the labels for the x, y, and z axes
ax.set_xlabel('X axis of the part (mm)', fontname='Arial Black', fontsize=20)
ax.set_ylabel('Layer #', fontname='Arial Black', fontsize=20)
ax.set_zlabel('Temperature (°C)', fontname='Arial Black', fontsize=20)

# Add a legend for the surface plot with larger font size
ax.legend([surf, surf_LPCtrl, surf_FRCtrl], ['No Ctrl', 'LP Ctrl', 'FR Ctrl'], loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=20)

# Rotate the plot to the right
ax.view_init(elev=10, azim=-30)

# Save the plot as .svg format with shrinked margin
# plt.savefig('/Users/zhangweijun/Documents/GitHub/Geng-code-FEM_ResidualStress/SVG_figures/Surface_STW_TemperatureComparison-Ro-30Deg.svg', bbox_inches='tight', format='svg')

# ***********************  3D Surface plot No_Ctrl & LP_Ctrl - END
plt.show()  # Show the plot








# #********************** Plot laser power, medium temperature for laser power-based energy ctrl vs. No Ctrl - START  
# # Load the Medium_temperature_RecSearch array from the file
# # Medium_temperature_RecSearch31x7 = np.load('Medium_temperature_RecSearch_7(row)x31(column).npy')
# # *************************************************************************

# plt.rcParams["font.weight"] = "bold"
# fig, ax1 = plt.subplots(figsize=(10, 8))  # Create a new figure with a custom size

# color_lp = 'indianred'
# # Plot the laser power on the left y-axis
# lns1 = ax1.plot(layer, lp_array[1:], linewidth=3, marker='.', markersize=10, color=color_lp, label='Laser power')
# ax1.set_ylabel('Laser Power (kW)', color=color_lp)
# ax1.tick_params(axis='y', labelcolor=color_lp, )
# ax1.set_xlabel("Layer #", fontweight='bold')
# ax1.set_ylabel("Laser Power (kW)", color=color_lp, fontweight='bold')


# # Create a second y-axis that shares the same x-axis
# ax2 = ax1.twinx()

# color_T = 'tab:blue'
# color_objT = 'blue'

# # Plot Medium_temperature_OriginalRectSearch against the layer number
# lns2 = ax2.plot(layer, np.full((Frame_index.shape[0], ), 1100),
#                 linestyle='dashed', linewidth=3, color=color_objT, label='Objective Temperature')
# lns3 = ax2.plot(layer, Medium_temperature_OriginalRectSearch, linewidth=3, marker='.', markersize=10, linestyle='-', color=color_T,label='Medium Temperature - LP Ctrl')
# # Plot Medium_temperature_RecSearch
# # lns4 = ax2.plot(layer, Medium_temperature_RecSearch, linewidth=3, marker='.', markersize=10, linestyle='-', color='green', label ='Medium Temperature - Rectangular Search 31 (column) x 11 (row) pixels')
# # lns4 = ax2.plot(layer, Medium_temperature_RecSearch31x7, linewidth=3, marker='.', markersize=10, linestyle='-', color='black', label ='Medium Temperature - No Ctrl')
# ax2.set_ylabel('Medium_temperature ($^\circ$C)', color=color_T, fontweight='bold')
# ax2.tick_params(axis='y', labelcolor=color_T)

# # Set up the labels for the x and y axes
# plt.xlabel('Layer Number #')
# plt.ylabel('Medium Temperature (°C)')

# # Add a legend
# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs, loc='lower center', framealpha=0.2)

# # Set up the title for the plot
# # plt.title('Medium temperature acqureied by different search methods')
# plt.title("No Ctrl Step Thin Wall_01312024", fontname="Arial Black",
#           size=15, fontweight="bold")

# # Add grid lines
# plt.grid(True)
# #********************** Plot medium temperature - END
# # plt.show()


# #********************** Plot captured Frame when laser head moves away - START
# # Create a new list to store the average values for each layer
# factor_px2mm_avePerlayer = []
# factor_px2mm_avePerlayer_List = np.array([])

# # Iterate over the indices in Frame_index - total 40 layers
# for index, frame_index_i in enumerate(Frame_index):
#     # Extract the frame at index i from Frame_history
#     frame = Frame_history[frame_index_i+1]  # In the original Geng's fuzzy ctrl code, the first FOUR element of Frame_index is [0, 0, 1, 2...], the first one is repeated, so here need to add 1 to frame_index_i

#     # Create a new figure with a custom size
#     fig, ax = plt.subplots()

#     imge_coord_perLayer, MaxT_RecSearch_perlayer,ic_MT_AFTER_RecSearch, y_i, y_f, N_Data_perLayer, Length_CurrLayer_mm = get_image_coord_ls(index+1,frame_index,Frame_history)

#     # Calculate the distance between each points in on layer in unit of pixels
#     print(f"The length of the current layer is: {Length_CurrLayer_mm} mm")
#     print(imge_coord_perLayer)
#     differences = np.diff(imge_coord_perLayer, axis=0)
#     print(f"The differences between each points in on layer #{index+1} in unit of pixels is: {differences}")

#     # Calculate the Euclidean distance between consecutive points
#     distances_px = np.sqrt(np.sum(differences**2, axis=1))
#     print(f"The Euclidean distance between consecutive points in on layer #{index+1} in unit of pixels is: {distances_px}")

#     distances_mm = (Length_CurrLayer_mm)/ (N_Data_perLayer - 1)
#     print(f"The distance between each points in on layer #{index+1} in unit of mm is: {distances_mm}")

#     # Calculate the average value of factor_px2mm for the current layer and # then append the average value to the factor_px2mm_avePerlayer list
#     factor_px2mm = distances_mm / distances_px
#     factor_px2mm_avePerlayer = np.mean(factor_px2mm)
#     factor_px2mm_avePerlayer_List = np.append(factor_px2mm_avePerlayer_List, np.mean(factor_px2mm))

#     print(f"The distance between each points in on layer #{index+1} in unit of pixels is: {distances_px}, and the factor_px2mm list is: {factor_px2mm}, the average factor_px2mm is: {factor_px2mm_avePerlayer} px/mm -- variant length (start from 10mm), 7 data points per layer")

#     print(f"For the layer #{index+1} , in captured frame #{frame_index_i+1}, the image coordinate used for calculation is:{imge_coord_perLayer} -- 31 x 5 symmetric Search")

#     # Add the total 8 image coordinates for a layer from the matrix transformation for each layer based on the IR camera 3Dto2D transformation calibration process
#     for j in imge_coord_perLayer:
#         center = tuple(j)
#         square_size = 1.0
#         center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
#         square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='yellow')  # Create a square
#         ax.add_patch(square)

#     # Add the max temperature point for each layer based on the rectangular search method
#     for k in ic_MT_AFTER_RecSearch:
#         center = tuple(k)
#         square_size = 1.0
#         center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
#         square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='green')  # Create a square
#         ax.add_patch(square)

#     # Add the max temperature point for each layer based on the origianl sqaure search method

#     # Add a legend
#     legend_elements = [
#         Patch(facecolor='yellow', edgecolor='none', label='Image Coordinates (Matrix Transformation)'),
#         Patch(facecolor='green', edgecolor='none', label='Max Temperature Point (31 x 11 Rectangular Search)'),
#     ]
#     ax.legend(handles=legend_elements, fontsize=9, bbox_to_anchor=(0, 0), loc='lower left')

#     # Display the first frame as an image
#     plt.imshow(frame, cmap='coolwarm', interpolation='nearest')

#     # Add a colorbar to the right of the plot
#     plt.colorbar(label='Value')

#     # Set up the title for the plot
#     plt.title(f'Layer #{index+1}')

# # After the loop, calculate the overall average
# overall_average = np.mean(factor_px2mm_avePerlayer_List)
# print(f"The overall average px2mm factor is: {overall_average} px/mm -- {distances_mm}mm length, {N_Data_perLayer}  data points per layer")




# # Show the plot
# # plt.show()
# #********************** Plot captured frame when laser head moves away - END