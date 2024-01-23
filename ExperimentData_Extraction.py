import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

with open('TW70_NCFR1000_Run20240121.npy', 'rb') as f:
    temp_layers = np.load(f, allow_pickle=True)
    laser_track = np.load(f, allow_pickle=True)
    lp_array = np.load(f, allow_pickle=True)
    fr_array = np.load(f, allow_pickle=True)
    temperature_history= np.load(f, allow_pickle=True)

    #Not sure if this is the Medium temperature data or sth else, I need to double check with Muqing
    # Medium_temperature = np.load(f, allow_pickle=True)
    # Medium_temperature = Medium_temperature[1:]

# Calculate the median temperature for each layer
Medium_temperature = np.median(temp_layers, axis=1)

# *** Plot Laser Power ***
plt.rcParams["font.weight"] = "bold"
layer = np.arange(1, temp_layers.shape[0]+1, dtype=int)
print(f'Temperature collected in each layer: {temp_layers}')
# print(f'laser_track: {laser_track}')
# print(f'Frame collected with shape: {temperature_history.shape}')
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
        plt.text(j, i, format(temp_layers[i, j], '.0f'),
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



# fig, ax1 = plt.subplots()
# color = 'indianred'
# lns1 = ax1.plot(layer, f_layer, linewidth=3, marker='.', markersize=10,
#                 color=color, label='Laser power')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xlabel("Layer #")
# ax1.set_ylabel("Laser Power (kW)", color=color, fontweight='bold')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Substrate Temperature ($^\circ$C)',
#                color=color, fontweight='bold')

# lns2 = ax2.plot(layer, np.full((50, ), 600),
#                 linestyle='dashed', linewidth=3, color='blue', label='Objective Temperature')
# # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# lns3 = ax2.plot(layer, median_temperature, color=color,
#                 linewidth=3, marker=".", markersize=10, label='Medium Temperature')
# ax2.tick_params(axis='y', labelcolor=color)

# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]

# plt.legend(lns, labs, loc='lower center', frameafha=0.2)
# plt.title("Laser Power-Based Energy Control Simulation", fontname="Arial Black",
#           size=15, fontweight="bold")
# ***********************

# # #*** Plot Feedrate ***
# plt.rcParams["font.weight"] = "bold"
# layer = np.arange(1, 51, dtype=np.int)
# print(fr_layer*60)
# fig, ax1 = plt.subplots()


# color = 'tab:purple'
# lns1 = ax1.plot(layer, fr_layer*60, linewidth=3, marker='.', markersize=10,
#                 color=color, label='Feed rate')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xlabel("Layer #", fontweight='bold')
# ax1.set_ylabel("Feed rate (mm/min)", color=color, fontweight='bold')


# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Substrate Temperature ($^\circ$C)',
#                color=color, fontweight='bold')
# lns2 = ax2.plot(layer, np.full((50, ), 600),
#                 linestyle='dashed', linewidth=3, color='blue', label='Objective Temperature')
# ax2.tick_params(axis='y', labelcolor=color)
# lns3 = ax2.plot(layer, median_temperature, color=color,
#                 linewidth=3, marker=".", markersize=10, label='Medium Temperature')
# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]

# plt.legend(lns, labs, loc='lower right', frameafha=0.2)

# ax2.plot()

# plt.title("Feed Rate-Based Energy Control Simulation", fontname="Arial Black",
#           size=15, fontweight="bold")
# #***********************

# # #*** Plot No Ctrl ***
# plt.rcParams["font.weight"] = "bold"
# layer = np.arange(1, 51, dtype=np.int)
# print(fr_layer*60)
# fig, ax1 = plt.subplots()
# color = 'tab:purple'
# lns1 = ax1.plot(layer, fr_layer*60, linewidth=2, marker='.', markersize=5,
#                 color=color, label='Feed rate (mm/min)')
# # ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xlabel("Layer #", fontweight='bold')
# ax1.set_ylabel("Deposition Parameters", fontweight='bold')
# color = 'indianred'
# lns2 = ax1.plot(layer, f_layer*1000, linewidth=2, marker='.', markersize=5,
#                 color=color, label='Laser power (W)')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Substrate Temperature ($^\circ$C)',
#                color=color, fontweight='bold')
# lns3 = ax2.plot(layer, np.full((50, ), 600),
#                 linestyle='dashed', linewidth=3, color='blue', label='Objective Temperature')
# lns4 = ax2.plot(layer, median_temperature, linewidth=3, marker='.',
#                 color=color, label='Medium Temperature')
# ax2.tick_params(axis='y', labelcolor=color)
# lns = lns1+lns2+lns3+lns4
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs, loc='center right', frameafha=0.2)

# plt.title("No Control Simulation", fontname="Arial Black",
#           size=15, fontweight="bold")


# # # ***********************

plt.show()
