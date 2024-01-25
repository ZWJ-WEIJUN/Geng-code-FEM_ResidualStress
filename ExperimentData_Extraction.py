import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle,Rectangle
from matplotlib.ticker import FormatStrFormatter



# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - START

def get_image_coord(Layer):
    x_start = 20.
    y_start = 8.
    z_start = -4.

    Layer_Height = 0.7
    Length = 70.

    L_Change = 0
    Layer_Change = 40
    N_Change = 1
    N_Data = 7
    Total_Layer = Layer_Change*N_Change

    Coord = [[0,0,0]]

    x_coord = x_start 
    y_coord = y_start 
    z_coord = z_start+Layer_Height

    for i in range(N_Change):
        x_i = x_start
        y_i = y_start + i*L_Change/2
        y_f = y_start + Length - i*L_Change/2
        z_i = z_start
        L_inc = (y_f-y_i)/N_Data
        for j in range(Layer_Change):
            x_coord = x_i
            y_coord = y_i
            Coord.append([x_coord,y_coord,z_coord])
            for z in range(N_Data):
                y_coord = y_coord + L_inc
                y_coord  = round(y_coord,3)
                Coord.append([x_coord,y_coord,z_coord])

            z_coord = round(z_coord+Layer_Height,3)
    Coord = Coord[1:]

    # Extract Image Coordinate
    Indexi = (Layer - 1) *N_Data+(Layer-1)
    Indexf= (Layer)*N_Data+Layer
    C_lt = Coord[Indexi:Indexf]
    C_lt = np.array(C_lt)


    date = '20240121'
    TranMat = np.load('TranMat_'+date+'.npy')
    C_lt = np.append(C_lt, np.array([1]*(len(C_lt))).reshape(-1,1), axis=1)

    image_coord =[]

    for mc in C_lt:
        ic = np.round(np.matmul(TranMat,mc)).astype(int)   
        image_coord.append(ic)
    image_coord = np.array(image_coord)
    
    return image_coord


# ************** Transformed pixel coordinates from IR camera 3Dto2D transformation calibration process - END





with open('TW70_NCFR1000_Run20240121.npy', 'rb') as f:
    temp_layers = np.load(f, allow_pickle=True)
    laser_track = np.load(f, allow_pickle=True)
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    Frame_history = np.load(f, allow_pickle=True)

    # Not sure if this is the Medium temperature data or sth else, I need to double check with Muqing
    Frame_index = np.load(f, allow_pickle=True)
    # Exclude the first element of Frame_index
    Frame_index = Frame_index[1:]
    # Medium_temperature = Medium_temperature[1:]

# Calculate the median temperature for each layer
Medium_temperature = np.median(temp_layers, axis=1)

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

plt.title('Temperature Distribution Layer by Layer')  # Add a title to the plot
#***********************  Heatmap plot - END


#********************** Plot medium temperature - START
plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
# Plot Medium_temperature against the layer number
plt.plot(layer, Medium_temperature, marker='o', linestyle='-', color='b')

# Set up the labels for the x and y axes
plt.xlabel('Layer Number #')
plt.ylabel('Medium Temperature (°C)')

# Set up the title for the plot
plt.title('Medium Temperature vs Layer Number')

# Add grid lines
plt.grid(True)
#********************** Plot medium temperature - END


#********************** Plot captured Frame - START
# Iterate over the indices in Frame_index
for index, frame_index_i in enumerate(Frame_index):
    print(frame_index_i)
    # Extract the frame at index i from Frame_history
    frame = Frame_history[frame_index_i+1]  # In the original Geng's fuzzy ctrl code, the first FOUR element of Frame_index is [0,0, 1, 2...], so I need to add 1 to frame_index_i

    # Create a new figure with a custom size
    fig, ax = plt.subplots()

    imge_coord = get_image_coord(index+1)

    print(f"For the layer #{index+1} , in captured frame #{frame_index_i+1}, the image coordinate used for calculation is:{imge_coord}")

    for j in imge_coord:
        center = tuple(j)
        square_size = 1 
        bottom_left_corner = (center[0] - square_size/2, center[1] - square_size/2)
        square = Rectangle(bottom_left_corner, square_size, square_size, color='yellow')  # Create a square
        ax.add_patch(square)

    # Display the first frame as an image
    plt.imshow(frame, cmap='coolwarm', interpolation='nearest')

    # Add a colorbar to the right of the plot
    plt.colorbar(label='Value')

    # Set up the title for the plot
    plt.title(f'Layer #{index+1}')





# Show the plot
plt.show()
#********************** Plot captured frame - END
