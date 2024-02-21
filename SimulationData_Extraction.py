import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# with open('thinwall_test_lp_control_lp1.7_04032023.npy', 'rb') as lp:
#     nodal_T2 = np.load(lp)
#     nodal_A2 = np.load(lp)
#     LSC = np.load(lp)
#     energy_lose = np.load(lp)
#     laser_energy = np.load(lp)
#     median_temperature = np.load(lp)
#     lp_layer = np.load(lp)
#     # fr_layer = np.load(lp)
#     dp_error = np.load(lp)
#     dp_dedl = np.load(lp)


with open('thinwall_test_fr_control_lp1.7_04042023.npy', 'rb') as f:
    nodal_T2 = np.load(f)   # nodal_T2 * nodal_volume * heat_capacity
    nodal_A2 = np.load(f)
    LSC = np.load(f)
    energy_lose = np.load(f)
    laser_energy = np.load(f)
    median_temperature = np.load(f)
    lp_layer = np.load(f)
    fr_layer = np.load(f)
    dp_error = np.load(f)
    dp_dedl = np.load(f)





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

# # *** Plot Laser Power ***
# plt.rcParams["font.weight"] = "bold"
# layer = np.arange(1, 51, dtype=np.int)
# print(lp_layer)
# fig, ax1 = plt.subplots()
# color = 'indianred'
# lns1 = ax1.plot(layer, lp_layer*1000, linewidth=3, marker='.', markersize=10,
#                 color=color, label='Laser power')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.set_xlabel("Layer #",  fontweight='bold')
# ax1.set_ylabel("Laser power (W)", color=color, fontweight='bold')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Temperature ($^\circ$C)',
#                color=color, fontweight='bold')

# lns2 = ax2.plot(layer, np.full((50, ), 600),
#                 linestyle='dashed', linewidth=3, color='blue', label='Objective temperature')
# # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# lns3 = ax2.plot(layer, median_temperature, color=color,
#                 linewidth=3, marker=".", markersize=10, label='Median temperature')
# ax2.tick_params(axis='y', labelcolor=color)

# lns = lns1+lns2+lns3
# labs = [l.get_label() for l in lns]

# plt.legend(lns, labs, loc='lower center', framealpha=0.2)
# # plt.title("Laser Power-Based Energy Control Simulation", fontname="Arial Black",
# #           size=15, fontweight="bold")
# # ***********************

# #*** Plot Feedrate ***
plt.rcParams["font.weight"] = "bold"
layer = np.arange(1, 51, dtype=np.int)
print(fr_layer*60)
fig, ax1 = plt.subplots()


color = 'tab:purple'
lns1 = ax1.plot(layer, fr_layer*60, linewidth=3, marker='.', markersize=10,
                color=color, label='Feed rate')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xlabel("Layer #", fontweight='bold')
ax1.set_ylabel("Feed rate (mm/min)", color=color, fontweight='bold')


ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Temperature ($^\circ$C)',
               color=color, fontweight='bold')
lns2 = ax2.plot(layer, np.full((50, ), 600),
                linestyle='dashed', linewidth=3, color='blue', label='Objective temperature')
ax2.tick_params(axis='y', labelcolor=color)
lns3 = ax2.plot(layer, median_temperature, color=color,
                linewidth=3, marker=".", markersize=10, label='Median temperature')
lns = lns1+lns2+lns3
labs = [l.get_label() for l in lns]

plt.legend(lns, labs, loc='lower right', framealpha=0.2)

# ax2.plot()

# plt.title("Feed Rate-Based Energy Control Simulation", fontname="Arial Black",
#           size=15, fontweight="bold")
# ***********************


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
# ax1.set_ylabel("Deposition parameters", fontweight='bold')
# color = 'indianred'
# lns2 = ax1.plot(layer, lp_layer*1000, linewidth=2, marker='.', markersize=5,
#                 color=color, label='Laser power (W)')

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Temperature ($^\circ$C)',
#                color=color, fontweight='bold')
# lns3 = ax2.plot(layer, np.full((50, ), 600),
#                 linestyle='dashed', linewidth=3, color='blue', label='Objective temperature')
# print(f'The median temperature for total 50 layer thinwall is {median_temperature}')
# lns4 = ax2.plot(layer, median_temperature, linewidth=3, marker='.',
#                 color=color, label='Median temperature')
# ax2.tick_params(axis='y', labelcolor=color)
# lns = lns1+lns2+lns3+lns4
# labs = [l.get_label() for l in lns]
# plt.legend(lns, labs, loc='center right', framealpha=0.2)

# # plt.title("No Control Simulation", fontname="Arial Black",
# #           size=15, fontweight="bold")


# # # ***********************

plt.show()
