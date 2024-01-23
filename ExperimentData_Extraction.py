import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# with open('thinwall_test_fr_control_lp1.7_04042023.npy', 'rb') as f:
#     nodal_T2 = np.load(f)   # nodal_T2 * nodal_volume * heat_capacity
#     nodal_A2 = np.load(f)
#     LSC = np.load(f)
#     energy_lose = np.load(f)
#     laser_energy = np.load(f)
#     median_temperature = np.load(f)
#     lp_layer = np.load(f)
#     fr_layer = np.load(f)
#     dp_error = np.load(f)
#     dp_dedl = np.load(f)


with open('TW70_NCFR1000_Run20240121.npy', 'rb') as lp:
    temp_layers = np.load(lp, allow_pickle=True)
    laser_track = np.load(lp, allow_pickle=True)
    lp_array = np.load(lp, allow_pickle=True)
    fr_array = np.load(lp, allow_pickle=True)
    temperature_history= np.load(lp, allow_pickle=True)
    frame_array= np.load(lp, allow_pickle=True)

# with open('thinwall_test_no_control_lp1.7_09162023.npy', 'rb') as f:
#     nodal_T2 = np.load(f)   # nodal_T2 * nodal_volume * heat_capacity
#     nodal_A2 = np.load(f)  # activate
#     LSC = np.load(f)
#     energy_lose = np.load(f)
#     laser_energy = np.load(f)
#     median_temperature = np.load(f)
#     lp_layer = np.load(f)
#     fr_layer = np.load(f)
#     dp_error = np.load(f)
#     dp_dedl = np.load(f)

# print(energy_lose)
# print(fr_layer)
# print(nodal_T2.size)

# Calculate the extent of the plot    
left = 0
right = temp_layers.shape[1]
bottom = 0
top = temp_layers.shape[0]


# *** Plot Laser Power ***
plt.rcParams["font.weight"] = "bold"
layer = np.arange(1, 41, dtype=int)
print(temp_layers)

temp_layers = np.array(temp_layers, dtype=float)

plt.figure(figsize=(10, 8))  # Create a new figure with a custom size
plt.imshow(temp_layers, cmap='coolwarm', interpolation='nearest', extent=[left, right, bottom, top],  origin='lower', aspect=0.4)  # Create a heatmap with the data
cbar =plt.colorbar(label='Temperature', shrink=0.8)  # Add a colorbar to the right of the plot
# cbar.ax.margins(y=0)  # Make the colorbar narrower

# Add numbers to each cell
for i in range(temp_layers.shape[0]):
    for j in range(temp_layers.shape[1]):
        plt.text(j+0.5, i+0.5, format(temp_layers[i, j], '.0f'),
                 horizontalalignment="center",
                 verticalalignment = "center",
                 color="white" if temp_layers[i, j] > temp_layers.max()/2 else "black", fontsize=8)

# Set up the labels for the x and y axes
plt.xlabel('Data Points in X axis')
plt.ylabel('Layers')

# Set up the ticks for the y axis
plt.yticks(ticks=np.arange(temp_layers.shape[0]), labels=np.arange(1, temp_layers.shape[0] + 1))

plt.title('Temperature Distribution Layers by Layer')  # Add a title to the plot
plt.show()  # Display the plot
# fig, ax1 = plt.subplots()
# color = 'indianred'
# lns1 = ax1.plot(layer, lp_layer, linewidth=3, marker='.', markersize=10,
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

# plt.legend(lns, labs, loc='lower center', framealpha=0.2)
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

# plt.legend(lns, labs, loc='lower right', framealpha=0.2)

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
# lns2 = ax1.plot(layer, lp_layer*1000, linewidth=2, marker='.', markersize=5,
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
# plt.legend(lns, labs, loc='center right', framealpha=0.2)

# plt.title("No Control Simulation", fontname="Arial Black",
#           size=15, fontweight="bold")


# # # ***********************

plt.show()
