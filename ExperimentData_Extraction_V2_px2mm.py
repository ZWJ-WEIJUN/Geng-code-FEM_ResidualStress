# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 17:42:50 2024
This code is used to extract the temperature data from the experiment IR data

@author: muqin/Weijun
"""
#V2, include local search function testing

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Rectangle
from matplotlib.ticker import FormatStrFormatter



# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - START

def get_image_coord(Layer,Frame_array,Frame_history):
    x_start = 20.
    y_start = 8.
    z_start = -4.

    Layer_Height = 0.7
    Length = 70.

    L_Change = 0
    Layer_Change = 40
    N_Change = 1
    N_Data_perLayer = 8          # number of data points collected in each layer
    Total_Layer = Layer_Change*N_Change

    Coord = [[0,0,0]]

    x_coord = x_start 
    y_coord = y_start 
    z_coord = z_start+Layer_Height

    # Search Parameters
    column=10
    row = 3

    
    for i in range(N_Change):
        x_i = x_start
        y_i = y_start + 5 + i*L_Change/2            # Here +5 is to make sure the first location for collecting data is not on the edge of the thin wall but at y = 13 mm
        y_f = y_start - 5 + Length - i*L_Change/2   # Here -5 is to make sure the last location for collecting data is not on the edge of the thin wall but at y = 73 mm   
        z_i = z_start     
        L_inc = (y_f-y_i)/(N_Data_perLayer - 1)
        for j in range(Layer_Change):
            x_coord = x_i
            y_coord = y_i
            Coord.append([x_coord,y_coord,z_coord])
            for z in range(N_Data_perLayer - 1):
                y_coord = y_coord + L_inc
                y_coord  = round(y_coord,3)
                Coord.append([x_coord,y_coord,z_coord])

            z_coord = round(z_coord+Layer_Height,3)
    Coord = Coord[1:]

    # Extract Image Coordinate
    Indexi = (Layer - 1) *(N_Data_perLayer - 1)+(Layer-1)
    Indexf= (Layer)*(N_Data_perLayer-1)+Layer
    C_lt = Coord[Indexi:Indexf]
    C_lt = np.array(C_lt)


    date = '20240121'
    TranMat = np.load('TranMat_'+date+'.npy')
    C_lt = np.append(C_lt, np.array([1]*(len(C_lt))).reshape(-1,1), axis=1)
    
    New_MT=[]  #max temperature from local search
    image_coord =[] 
    Search_coord = [] #the search coord
    # Index_test=[]
    frame_index = Frame_array[Layer]
    temperature_data=Frame_history[frame_index+1]
    for mc in C_lt:
        ic = np.round(np.matmul(TranMat,mc)).astype(int)   
        image_coord.append(ic)
        
        local_search = temperature_data[ic[1]-row:ic[1]+row+1,ic[0]-column:ic[0]+column+1]
        if np.any(local_search):
            local_high_temp = np.max(local_search)
            index = np.unravel_index(np.argmax(local_search), np.shape(local_search))
            max_coord = [index[1]+ic[0]-column,index[0]+ic[1]-row]  #the coordinate of the max temperature in the local search area
        else:
            local_high_temp = temperature_data[ic[1],ic[0]]
        New_MT.append(local_high_temp)
        Search_coord.append(max_coord)
        # Index_test.append(index)
   
    image_coord = np.array(image_coord)  #
    New_MT = np.array(New_MT)            # Max temperature acquired based on new local search method
    Search_coord = np.array(Search_coord) # The coordinate of the max temperature in the local search area
    # Index_test=np.array(Index_test)
    
    return image_coord, New_MT, Search_coord, y_i, y_f, N_Data_perLayer


# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - END





with open('TW70_NCFR1000_Run20240121.npy', 'rb') as f:
    temp_layers = np.load(f, allow_pickle=True)
    laser_track = np.load(f, allow_pickle=True)
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True)

    # Not sure if this is the Medium temperature data or sth else, I need to double check with Muqing
    frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index
    Frame_index = frame_index[1:]
    # Medium_temperature = Medium_temperature[1:]

# Calculate the median temperature for each layer
Medium_temperature = np.median(temp_layers, axis=1)


#Create new temperature distribution with new methond
New_temp_layer =[]
for i in range (1,len(frame_index)):
   imge_coord_perLayer, New_MT,Search_coord, y_i, y_f, N_Data_perLayer = get_image_coord(i,frame_index,Frame_history)
   New_temp_layer.append(New_MT)

# *** Plot Laser Power ***
plt.rcParams["font.weight"] = "bold"
layer = np.arange(1, temp_layers.shape[0]+1, dtype=int)
print(f'Temperature collected in each layer: {temp_layers}')
print(f'laser_track: {laser_track}')
print(f'Frame collected (np.ndarray): {Frame_history.shape}')
print(f'Frame index: {Frame_index}')

print(f'Medium temperature: {Medium_temperature}')

# Extract the first sublist from laser_track
first_sublist = laser_track[0]
# Extract the second element from each sublist in first_sublist
second_elements = [sublist[1] for sublist in first_sublist]

x_labels = np.array(second_elements, dtype=int)
temp_layers = np.array(temp_layers, dtype=float)

# ***********************  Heatmap plot - START
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
plt.imshow(temp_layers, cmap='coolwarm', interpolation='nearest',  origin='lower', aspect=0.4)  # Create a heatmap with the data
cbar =plt.colorbar(label='Temperature (°C)', shrink=0.8)  # Add a colorbar to the right of the plot
# cbar.ax.margins(y=0)  # Make the colorbar narrower

# Add numbers to each cell
for i in range(temp_layers.shape[0]):
    for j in range(temp_layers.shape[1]):
        plt.text(j, i, format(temp_layers[i, j], '.2f'),
                 horizontalalignment="center",
                 verticalalignment = "center",
                 color="white" if temp_layers[i, j] > 900 else "black", fontsize=8)

# Set up the labels for the x and y axes
plt.xlabel('Coordinate along machine Y axis (mm)')
plt.ylabel('Layers #')

# Set up the ticks for the y axis
plt.yticks(ticks=np.arange(temp_layers.shape[0]), labels=np.arange(1, temp_layers.shape[0] + 1)) # temp_layers.shape[0] = 40
# Set up the ticks for the x axis
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)

plt.title('Temperature Distribution:Square Search')  # Add a title to the plot
#***********************  Heatmap plot - END


New_temp_layer = np.array(New_temp_layer, dtype=float)

# ***********************  Heatmap plot_Rectangular Search new figure with a custom size
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
plt.imshow(New_temp_layer, cmap='coolwarm', interpolation='nearest',  origin='lower', aspect=0.4)  # Create a heatmap with the data
cbar =plt.colorbar(label='Temperature (°C)', shrink=0.8)  # Add a colorbar to the right of the plot
# cbar.ax.margins(y=0)  # Make the colorbar narrower

# Add numbers to each cell
for i in range(New_temp_layer.shape[0]):
    for j in range(New_temp_layer.shape[1]):
        plt.text(j, i, format(New_temp_layer[i, j], '.2f'),
                 horizontalalignment="center",
                 verticalalignment = "center",
                 color="white" if temp_layers[i, j] > 900 else "black", fontsize=8)

# Set up the labels for the x and y axes
plt.xlabel('Coordinate along machine Y axis (mm)')
plt.ylabel('Layers #')

# Set up the ticks for the y axis
plt.yticks(ticks=np.arange(temp_layers.shape[0]), labels=np.arange(1, temp_layers.shape[0] + 1)) # temp_layers.shape[0] = 40
# Set up the ticks for the x axis
plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels)

plt.title('Temperature Distribution:Rectangular Search')  # Add a title to the plot
#***********************  Heatmap plot - END



#********************** Plot medium temperature - START
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# Plot Medium_temperature against the layer number
plt.plot(layer, Medium_temperature, marker='o', linestyle='-', color='b')

# Set up the labels for the x and y axes
plt.xlabel('Layer Number #')
plt.ylabel('Medium Temperature (°C)')

# Set up the title for the plot
plt.title('Medium Temperature vs Layer Number - Non non-symmetric Square Search (8 x 8)')

# Add grid lines
plt.grid(True)
#********************** Plot medium temperature - END


#********************** Plot captured Frame when laser head moves away - START
# Create a new list to store the average values for each layer
factor_px2mm_avePerlayer = []
factor_px2mm_avePerlayer_List = np.array([])

# Iterate over the indices in Frame_index
for index, frame_index_i in enumerate(Frame_index):
    print(frame_index_i)
    # Extract the frame at index i from Frame_history
    frame = Frame_history[frame_index_i+1]  # In the original Geng's fuzzy ctrl code, the first FOUR element of Frame_index is [0, 0, 1, 2...], so I need to add 1 to frame_index_i

    # Create a new figure with a custom size
    fig, ax = plt.subplots()

    imge_coord_perLayer, New_MT,Search_coord, y_i, y_f, N_Data_perLayer = get_image_coord(index+1,frame_index,Frame_history)

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


    for j in imge_coord_perLayer:
        center = tuple(j)
        square_size = 1.0
        center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
        square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='yellow')  # Create a square
        ax.add_patch(square)
    
    for k in Search_coord:
        center = tuple(k)
        square_size = 1.0
        center_pt_of_sqaure = (center[0] - square_size/2, center[1] - square_size/2)
        square = Rectangle(center_pt_of_sqaure, square_size, square_size, color='green')  # Create a square
        ax.add_patch(square)

    # Display the first frame as an image
    plt.imshow(frame, cmap='coolwarm', interpolation='nearest')

    # Add a colorbar to the right of the plot
    plt.colorbar(label='Value')

    # Set up the title for the plot
    plt.title(f'Layer #{index+1}')

# After the loop, calculate the overall average
overall_average = np.mean(factor_px2mm_avePerlayer_List)
print(f"The overall average px2mm factor is: {overall_average} px/mm -- 10mm length, 8 data points per layer")




# Show the plot
plt.show()
#********************** Plot captured frame when laser head moves away - END
