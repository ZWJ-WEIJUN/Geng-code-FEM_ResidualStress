# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:21:27 2024

@author: Muqing
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.close('all')

# Define the dp_error, dp_derrordl, and dp_lp as per the code's structure for laser power control
dp_error = [[-50, -25, 0, 25, 50]]
dp_derrordl = [[-30, -15, 0, 15, 30]]
dp_lp = [[-80, -40, 0, 30, 60]]

# Simulate the fuzzy control function for a grid of inputs
def simulate_fuzzy_control(lp_current, obj_temp, activation_threshold, stable_threshold, dp_error, dp_derrordl, dp_lp):
    e_values = np.linspace(-50, 50, 50)  # error values range
    de_dl_values = np.linspace(-30, 30, 30)  # derrordl values range
    
    # Create a meshgrid for inputs (error, derrordl)
    E, dE_dl = np.meshgrid(e_values, de_dl_values)
    lp_incr_surface = np.zeros_like(E)
    
    # Iterate over the meshgrid to calculate lp_incr based on the fuzzy rules
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            # Simulate fuzzy control for laser power increment (lp_incr)
            error_p = [0., 0., 0., 0., 0.]
            derrordl_p = [0., 0., 0., 0., 0.]

            error = E[i, j]
            derrordl = dE_dl[i, j]

            # Calculate the membership probabilities for error
            if(dp_error[0][0] >= error):
                error_p[0] = 1.0
            elif(dp_error[0][4] <= error):
                error_p[4] = 1.0
            else:
                k = 0
                while error > dp_error[0][k]:
                    if error <= dp_error[0][k + 1]:
                        error_p[k] = (dp_error[0][k + 1] - error) / (dp_error[0][k + 1] - dp_error[0][k])
                        error_p[k + 1] = 1 - error_p[k]
                        break
                    else:
                        k += 1

            # Calculate the membership probabilities for derrordl
            if(dp_derrordl[0][0] >= derrordl):
                derrordl_p[0] = 1.0
            elif(dp_derrordl[0][4] <= derrordl):
                derrordl_p[4] = 1.0
            else:
                k = 0
                while derrordl > dp_derrordl[0][k]:
                    if derrordl <= dp_derrordl[0][k + 1]:
                        derrordl_p[k] = (dp_derrordl[0][k + 1] - derrordl) / (dp_derrordl[0][k + 1] - dp_derrordl[0][k])
                        derrordl_p[k + 1] = 1 - derrordl_p[k]
                        break
                    else:
                        k += 1

            # Apply the fuzzy inference rules and calculate the control_rules matrix
            control_rules = [[min(error_p[m], derrordl_p[n]) for n in range(5)] for m in range(5)]

            # Defuzzification (center of gravity method)
            area_sum = 0.0
            weighted_area = 0.0
            base = abs(dp_lp[0][4] - dp_lp[0][2])  # Laser power control (LP)
            for m in range(5):
                for n in range(5):
                    applied_rule = max(0, min(m + n - 3, 4))  # Determine the applied fuzzy set
                    area = (2 - control_rules[m][n]) * control_rules[m][n] * base / 2
                    area_sum += area
                    weighted_area += area * dp_lp[0][applied_rule]

            # Store the lp_incr value for this (error, derrordl) pair
            if area_sum != 0:
                lp_incr_surface[i, j] = weighted_area / area_sum
            else:
                lp_incr_surface[i, j] = 0

    return E, dE_dl, lp_incr_surface
plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.family"] = "Times New Roman"
plt.rc('font', weight='bold')
# Simulate the fuzzy controller's behavior for laser power increment
E, dE_dl, lp_incr_surface = simulate_fuzzy_control(0, 25, 1500, [1, 1], dp_error, dp_derrordl, dp_lp)

# Plot the surface diagram for laser power control mode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(E, dE_dl, lp_incr_surface, cmap='viridis')
# Make tick labels bold
# for tick in ax.xaxis.get_major_ticks():
#     tick.label.set_fontweight('bold')
# for tick in ax.yaxis.get_major_ticks():
#     tick.label.set_fontweight('bold')
# for tick in ax.zaxis.get_major_ticks():
#     tick.label.set_fontweight('bold')
# # Label the axes and add title
ax.set_xlabel('$e$ (°C)', fontweight="bold",size=15)
ax.set_ylabel('$\Delta$ T (°C)', fontweight="bold",size=15)
# ax.set_zlabel('LP Increment (W)', fontweight="bold",size=15)
ax.set_zlabel(r'$\Delta$ LP (W)', fontweight="bold",size=15)
# ax.set_title('Fuzzy Control: Laser Power Increment Surface')
ax.tick_params(axis='both', labelcolor='black',labelsize=15)

# Show color bar
# Create an inset axes for the colorbar
cax = ax.inset_axes([1.1, 0.15, 0.05, 0.7])  # [left, bottom, width, height]

# Add a colorbar to the inset axes
cbar = plt.colorbar(surf,cax)
cbar.ax.tick_params(labelsize=15) 
plt.subplots_adjust(left=0, right=0.7, top=0.95, bottom=0.15)
plt.tight_layout()
# Display the plot

plt.show()
