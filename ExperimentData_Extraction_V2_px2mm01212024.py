# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:42:50 2024
This code is used to extract the temperature data from the experiment IR data
V2: Recrtangular search testing and data extraction of original data from experiment 
V3: Shifting method base on maximum tempearture and data extraction of original data from experiment

@author: muqin/Weijun
"""
#V2, include local search function testing

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Rectangle,Patch
from matplotlib.ticker import FormatStrFormatter





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
    MT_RecSearch_perlayer: the max temperature acquired based on new local search method -  rectangular search method - for 8 points in each layer
    """
    x_start = 20.
    y_start = 8.
    z_start = -4.          # Here use -4 is due to the 4mm thickness of the calibration black gasket plate on the top of the Z0 substrate`

    Layer_Height = 0.7
    Length = 70.                # length of the thin wall in the y direction

    L_Change = 0                 # change of the length for step thin wall in the y direction
    Layer_Change = 40            # number of layers
    N_Change = 1                 # number of changes of the length for step thin wall in the y direction
    N_Data_perLayer = 8          # number of data points collected in each layer
    Total_Layer = Layer_Change*N_Change  # total number of layers calculation for the step thin wall

    Coord_TW_3D = []

    x_coord = x_start 
    y_coord = y_start 
    z_coord = z_start+Layer_Height   
    """??? 1. Don't understand why -4 + 0.7 = -3.3 ???"""



    # For look here is to get the coordinate of the thin wall geometry in the 3D printing space
    # Total 320 data points, 8 data points per layer, 40 layers
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
        # Coord_TW_3D - a list contains 320 data points, 8 data points per layer, 40 layers

    # Extract Image Coordinate
    Indexi = (Layer_num - 1) *(N_Data_perLayer - 1)+(Layer_num-1)   # Indexi is the start index of the data points for the current layer
    Indexf= (Layer_num)*(N_Data_perLayer-1)+Layer_num # Indexf is the end index of the data points for the current layer
    Coor_CurrLayer = Coord_TW_3D[Indexi:Indexf]                       # Coor_CurrLayer is the list of the 8 eight coordinate for the current layer
                                                            # The shape of Coord_TW_3D is: (320, 3)
    Coor_CurrLayer = np.array(Coor_CurrLayer)                             # Convert Coor_CurrLayer to a numpy array


    date = '20240121'
    TranMat = np.load('TranMat_'+date+'.npy')    # Size 2 x 4, 2 rows and 4 columns
    # print(f"{TranMat.shape} is the shape of the TranMat matrix.")

    Coor_CurrLayer = np.append(Coor_CurrLayer, np.array([1]*(len(Coor_CurrLayer))).reshape(-1,1), axis=1) # Add a column of 1 to the Coor_CurrLayer array
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
    MT_RecSearch_perlayer = []                                           # max temperature from rectangular local search
    ic_MT_AFTER_RecSearch = []                                   # the search coord
    MT_SquareSearch_perlayer = []                                           # max temperature from square local search
    ic_MT_AFTER_SquareSearch = []                                   # the search coord

    # Define the size of the rectangualr search area
    column = 10
    row = 3
    # Define the size of the square for local search
    square_size = 4
  
    Frame_index_per_layer = Frame_index[Layer_num]
    temperature_data = Frame_history[Frame_index_per_layer+1] # In the original Geng's fuzzy ctrl code, the first FOUR element of Frame_index is [0, 0, 1, 2...], the first one is repeated, so here need to add 1 to frame_index_i
    # temperature_data is the temperature data for the each layer, shape is (288, 382) - Please note that 288 rows (y) and 382 columns (x)

    # For loop here is to get the 2D image coordinate of the thin wall geometry in the 3D printing space for each point in each layer
    for mc in Coor_CurrLayer:
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
        MT_RecSearch_perlayer.append(local_high_temp)
        ic_MT_AFTER_RecSearch.append(max_temp_coord)    # Append the coordinate of the max temperature in the local search area to the ic_MT_AFTER_RecSearch list

        local_search_Square = temperature_data[ic[1]-square_size:ic[1]+square_size,ic[0]-square_size:ic[0]+square_size]          # local search square area of 8x8 pixels (8columns x 8rows)
        if np.any(local_search_Square):
            local_high_temp_Square = np.max(local_search_Square)                                                      # the max temperature in the local search area
            index_local_search_Square = np.unravel_index(np.argmax(local_search_Square), np.shape(local_search_Square))                   # the index of the max temperature in the local search area in the form of tuple with two elements (x,y) or (column, row)
            max_temp_coord_Square = [index_local_search_Square[1]+ic[0]-square_size,index_local_search_Square[0]+ic[1]-square_size]                         # max_temp_coord is the coordinate of the max temperature in the big image frame after locaL serach, not in the local search area
        else:
            local_high_temp_Square = temperature_data[ic[1],ic[0]]      #If the local search area is empty, then the max temperature is the temperature at the current point in the big image frame
            """Here is confusing, the temperature_data is 288rows by 382columns, if the origin is the top left corner of the image frame, the x axis is the horizontal axis(column), the y axis is the vertical axis (row),
            so the temepratre data is y by x, but the varaible ic is x (column) by y(row), so the temperature_data[ic[1],ic[0]] is the correct way to get the temperature at the current point in the big image frame"""
        MT_SquareSearch_perlayer.append(local_high_temp_Square)
        ic_MT_AFTER_SquareSearch.append(max_temp_coord_Square)    # Append the coordinate of the max temperature in the local search area to the ic_MT_AFTER_RecSearch list

   
    image_coord_ls = np.array(image_coord_ls)          # Convert the image_coord_ls list to a numpy array
    MT_RecSearch_perlayer = np.array(MT_RecSearch_perlayer)            # Max temperature acquired based on rectangular local search method for 8 points in each layer
    ic_MT_AFTER_RecSearch = np.array(ic_MT_AFTER_RecSearch)           # The coordinate of the max temperature in the local rectangular search area for 8 points in each layer
    MT_SquareSearch_perlayer = np.array(MT_SquareSearch_perlayer)            # Max temperature acquired based on suqare local search method for 8 points in each layer
    ic_MT_AFTER_SquareSearch = np.array(ic_MT_AFTER_SquareSearch)           # The coordinate of the max temperature in the local square search area for 8 points in each layer
    # Index_test=np.array(Index_test)
    
    return image_coord_ls, MT_RecSearch_perlayer, ic_MT_AFTER_RecSearch, ic_MT_AFTER_SquareSearch, y_i, y_f, N_Data_perLayer


# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - END





with open('TW70_NCFR1000_Run20240121.npy', 'rb') as f:
    MT_OriginSquareSearch_alllayers = np.load(f, allow_pickle=True)  # MT_OriginSquareSearch_alllayers is a np.ndarray that contains the temperature data collected by the IR camera during the experiment, shape is (40, 8)
    Coord_TW_3D_40layers = np.load(f, allow_pickle=True) # Coord_TW_3D_40layers is the thin wall coordinate in 3D, shape is (40, 8, 3)  
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True) # Frame_history is a np.ndarray that contains all the frames collected by the IR camera during the experiment

    # frame index is the index of the frame that collected by the IR camera by the laser head moves away from the thin wall
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index because the first element is ZERO, which is not a valid index for the Frame_history.
    Frame_index = frame_index[1:] # Frame_index is the index of the frame 1-40 that collected by the IR camera by the laser head moves away from the thin wall

# Calculate the median temperature for each layer
Medium_temperature_SqaureSearch = np.median(MT_OriginSquareSearch_alllayers, axis=1)   # Median temperature for each layer based on the non-symmetric square search method
# When axis = 1, it means that the medians will be computed along each row of the input array. In other words, the function will calculate the median value for each row separately.



# Create new temperature distribution with new rectangualr search methond
MT_RecSearch_alllayers =[]
for layer_num in range (1,len(frame_index)):
   imge_coord_perLayer, MT_RecSearch_perlayer,ic_MT_AFTER_RecSearch, ic_MT_AFTER_SquareSearch, y_i, y_f, N_Data_perLayer = get_image_coord_ls(layer_num,frame_index,Frame_history)
   MT_RecSearch_alllayers.append(MT_RecSearch_perlayer)

Medium_temperature_RecSearch = np.median(MT_RecSearch_alllayers, axis=1)   # Median temperature for each layer based on the new local search method - rectangular search method


# *** Plot Laser Power ***
plt.rcParams["font.weight"] = "bold"
layer = np.arange(1, MT_OriginSquareSearch_alllayers.shape[0]+1, dtype=int)
print(f'Frame collected (np.ndarray): {Frame_history.shape}')
print(f'Frame index: {Frame_index}')

print(f'Medium temperature based on origianl non-symmetric square search method: {Medium_temperature_SqaureSearch}')

# Extract the first sublist from Coord_TW_3D_40layers
first_sublist = Coord_TW_3D_40layers[0]
# Extract the second element from each sublist in first_sublist
second_elements = [sublist[1] for sublist in first_sublist]

x_labels = np.array(second_elements, dtype=int)
MT_OriginSquareSearch_alllayers = np.array(MT_OriginSquareSearch_alllayers, dtype=float)

# ***********************  Heatmap plot - START
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
plt.imshow(MT_OriginSquareSearch_alllayers, cmap='coolwarm', interpolation='nearest',  origin='lower', aspect=0.4)  # Create a heatmap with the data
cbar =plt.colorbar(label='Temperature (°C)', shrink=0.8)  # Add a colorbar to the right of the plot
# cbar.ax.margins(y=0)  # Make the colorbar narrower

# Add numbers to each cell
for i in range(MT_OriginSquareSearch_alllayers.shape[0]):
    for j in range(MT_OriginSquareSearch_alllayers.shape[1]):
        plt.text(j, i, format(MT_OriginSquareSearch_alllayers[i, j], '.2f'),
                 horizontalalignment="center",
                 verticalalignment = "center",
                 color="white" if MT_OriginSquareSearch_alllayers[i, j] > 900 else "black", fontsize=8)

# Set up the labels for the x and y axes
plt.xlabel('Coordinate along machine Y axis (mm)')
plt.ylabel('Layers #')

# Set up the ticks for the y axis
plt.yticks(ticks=np.arange(MT_OriginSquareSearch_alllayers.shape[0]), labels=np.arange(1, MT_OriginSquareSearch_alllayers.shape[0] + 1)) # MT_OriginSquareSearch_alllayers.shape[0] = 40
# Set up the ticks for the x axis
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)

plt.title('Temperature Distribution - Square Search')  # Add a title to the plot
#***********************  Heatmap plot - END


MT_RecSearch_alllayers = np.array(MT_RecSearch_alllayers, dtype=float)

# ***********************  Heatmap plot_Rectangular Search new figure with a custom size
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
plt.imshow(MT_RecSearch_alllayers, cmap='coolwarm', interpolation='nearest',  origin='lower', aspect=0.4)  # Create a heatmap with the data
cbar =plt.colorbar(label='Temperature (°C)', shrink=0.8)  # Add a colorbar to the right of the plot
# cbar.ax.margins(y=0)  # Make the colorbar narrower

# Add numbers to each cell
for i in range(MT_RecSearch_alllayers.shape[0]):
    for j in range(MT_RecSearch_alllayers.shape[1]):
        plt.text(j, i, format(MT_RecSearch_alllayers[i, j], '.2f'),
                 horizontalalignment="center",
                 verticalalignment = "center",
                 color="white" if MT_OriginSquareSearch_alllayers[i, j] > 900 else "black", fontsize=8)

# Set up the labels for the x and y axes
plt.xlabel('Coordinate along machine Y axis (mm)')
plt.ylabel('Layers #')

# Set up the ticks for the y axis
plt.yticks(ticks=np.arange(MT_OriginSquareSearch_alllayers.shape[0]), labels=np.arange(1, MT_OriginSquareSearch_alllayers.shape[0] + 1)) # MT_OriginSquareSearch_alllayers.shape[0] = 40
# Set up the ticks for the x axis
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)

plt.title('Temperature Distribution - Rectangular Search')  # Add a title to the plot
#***********************  Heatmap plot - END



#********************** Plot medium temperature - START
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# Plot Medium_temperature_SqaureSearch against the layer number
plt.plot(layer, Medium_temperature_SqaureSearch, marker='o', linestyle='-', color='purple', label='Square Search Non-symmetric 8 x 8 pixels')
# Plot Medium_temperature_RecSearch
plt.plot(layer, Medium_temperature_RecSearch, marker='o', linestyle='-', color='green', label ='Rectangular Search 21 x 7 pixels')

# Set up the labels for the x and y axes
plt.xlabel('Layer Number #')
plt.ylabel('Medium Temperature (°C)')

# Add a legend
plt.legend()

# Set up the title for the plot
plt.title('Medium temperature acqureied by different search methods')



# Add grid lines
plt.grid(True)
#********************** Plot medium temperature - END


#********************** Plot captured Frame when laser head moves away - START
# Create a new list to store the average values for each layer
factor_px2mm_avePerlayer = []
factor_px2mm_avePerlayer_List = np.array([])

# Iterate over the indices in Frame_index - total 40 layers
for index, frame_index_i in enumerate(Frame_index):
    print(f"The frame index number is: {frame_index_i}")
    # Extract the frame at index i from Frame_history
    frame = Frame_history[frame_index_i+1]  # In the original Geng's fuzzy ctrl code, the first FOUR element of Frame_index is [0, 0, 1, 2...], the first one is repeated, so here need to add 1 to frame_index_i

    # Create a new figure with a custom size
    fig, ax = plt.subplots()

    imge_coord_perLayer, MT_RecSearch_perlayer,ic_MT_AFTER_RecSearch, ic_MT_AFTER_SquareSearch, y_i, y_f, N_Data_perLayer = get_image_coord_ls(index+1,frame_index,Frame_history)

    # Calculate the distance between each points in on layer in unit of pixels
    differences = np.diff(imge_coord_perLayer, axis=0)

    # Calculate the Euclidean distance between consecutive points
    distances_px = np.sqrt(np.sum(differences**2, axis=1))

    distances_mm = (y_f-y_i)/ (N_Data_perLayer - 1)

    # Calculate the average value of factor_px2mm for the current layer and # then append the average value to the factor_px2mm_avePerlayer list
    factor_px2mm = distances_mm / distances_px
    factor_px2mm_avePerlayer = np.mean(factor_px2mm)
    factor_px2mm_avePerlayer_List = np.append(factor_px2mm_avePerlayer_List, np.mean(factor_px2mm))

    print(f"The distance between each points in on layer #{index+1} in unit of pixels is: {distances_px}, and the factor_px2mm list is: {factor_px2mm}, the average factor_px2mm is: {factor_px2mm_avePerlayer} px/mm -- 10mm length, 8 data points per layer")

    print(f"For the layer #{index+1} , in captured frame #{frame_index_i+1}, the image coordinate used for calculation is:{imge_coord_perLayer} -- 8 x 8 Non-symmetric Search")

    # Add the total 8 image coordinates for a layer from the matrix transformation for each layer based on the IR camera 3Dto2D transformation calibration process
    for j in imge_coord_perLayer:
        center = tuple(j)
        square_size = 1.0
        center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
        square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='yellow')  # Create a square
        ax.add_patch(square)

    # Add the max temperature point for each layer based on the rectangular search method
    for k in ic_MT_AFTER_RecSearch:
        center = tuple(k)
        square_size = 1.0
        center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
        square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='green')  # Create a square
        ax.add_patch(square)

    # Add the max temperature point for each layer based on the origianl sqaure search method
    for l in ic_MT_AFTER_SquareSearch:
        center = tuple(l)
        square_size = 1.0
        center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
        square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='purple')  # Create a square
        ax.add_patch(square)

    # Add a legend
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='none', label='Image Coordinates (Matrix Transformation)'),
        Patch(facecolor='green', edgecolor='none', label='Max Temperature Point (Rectangular Search)'),
        Patch(facecolor='purple', edgecolor='none', label='Max Temperature Point (Square Search)')
    ]
    ax.legend(handles=legend_elements, fontsize=9, bbox_to_anchor=(0, 0), loc='lower left')

    # Display the first frame as an image
    plt.imshow(frame, cmap='coolwarm', interpolation='nearest')

    # Add a colorbar to the right of the plot
    plt.colorbar(label='Value')

    # Set up the title for the plot
    plt.title(f'Layer #{index+1}')

# After the loop, calculate the overall average
overall_average = np.mean(factor_px2mm_avePerlayer_List)
print(f"The overall average px2mm factor is: {overall_average} px/mm -- {distances_mm}mm length, 8 data points per layer")




# Show the plot
plt.show()
#********************** Plot captured frame when laser head moves away - END
