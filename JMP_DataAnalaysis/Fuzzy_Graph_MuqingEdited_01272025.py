# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:05:29 2024

@author: muqin
"""
import numpy as np
import fuzzylab as fl
import matplotlib.pyplot as plt
plt.close('all')

plt.rcParams["font.weight"] = "normal"
plt.rcParams["font.family"] = "Times New Roman"
def fuzzy_dp(dp, dp_name,graph_setting):
    x = fl.arange(dp[0]-(dp[-1]-dp[2]), 0.1,dp[-1]+(dp[-1]-dp[2]))
    y = [1,1,1,1,1]
    # Lowerest = fl.trimf(x, [dp[0]-abs(dp[-1]), dp[1], dp[2]])
    Lowest = fl.trapmf(x, [dp[0]-200,dp[0]-100, dp[0], dp[1]])
    Lower = fl.trimf(x, [dp[0], dp[1], dp[2]])
    Mid = fl.trimf(x, [dp[1], dp[2], dp[3]])
    Higher = fl.trimf(x, [dp[2], dp[3], dp[4]])
    Highest = fl.trapmf(x, [dp[3],dp[4], dp[4]+100, dp[4]+200])
    fig, ax1 = plt.subplots(figsize=(10, 8))
    # plt.title(graph_setting[0],fontname="Times New Roman",
    #           size=15, fontweight="bold")
    lns1 = plt.plot(x,Lowest,label=dp_name[0],linewidth=3)
    lns2 = plt.plot(x,Lower,label=dp_name[1],linewidth=3)
    lns3 = plt.plot(x,Mid,label=dp_name[2],linewidth=3)
    lns4 = plt.plot(x,Higher,label=dp_name[3],linewidth=3)
    lns5 = plt.plot(x,Highest,label=dp_name[4],linewidth=3)
    lns = lns1+lns2+lns3+lns4+lns5
    for a,b in zip(dp, y): 
        plt.text(a, b, str(a),ha='center', va='bottom', fontsize=30, fontweight="bold")
    plt.ylim([-0.55, 1.2])
    plt.xlim([dp[0]-(dp[-1]-dp[2]), dp[-1]+(dp[-1]-dp[2])])
    plt.xlabel(graph_setting[1], fontweight="bold",size=30)
    plt.ylabel('Probability', fontweight="bold",size=30)
    plt.ylabel(r'$\mu$', fontweight="bold",size=30)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax1.tick_params(axis='both', labelcolor='black',labelsize=30)
    # Make tick labels bold for ax1
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    
    
    labs = [l.get_label() for l in lns]
    legend = plt.legend(lns, labs, loc='lower center', framealpha=0.5, ncol=2,fontsize=23.5)
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.tight_layout() 
    plt.subplots_adjust(left=0.18, right=0.88, top=0.95, bottom=0.15)
    # Highest = fl.trimf(x, [dp[2], dp[3], dp[4]+abs(dp[-1])])
    return x, Lowest,Lower,Mid,Higher,Highest

dp_error    = [-50,-25,0,+25,+50]
dp_derrordl = [-30,-15,0,15,30]
dp_fr       = [-1,-.5,0,.5,1]
dp_lp       = [0.06,0.03,0,-0.04,-0.08]  
dp_lp       = [-80,-40,0,+30,+60]  

dp_error_name=['Large-negative', 'Small-negative', 'Zero','Small-positive','Large-positive']
dp_error_graph= [' ', 'e (°C)']
# dp_error_name=['Negative-large', 'Negative-small', 'Zero','Positive-small','Positive-large']
# dp_error_graph= [' ', '$e$  (°C)']

dp_derrordl_name=['Large Negative', 'Negative', 'Zero','Positive','Large Positive']
dp_derrordl_graph= ['Rate of Median Median Temperature Change', 'Temperature Change (°C)']


dp_fr_name=['Large Negative', 'Negative', 'Zero','Positive','Large Positive']
dp_fr_graph= ['FR Increment Membership Function', 'Feed Rate (mm/s)']


dp_lp_name=['Ln', 'Sn', 'Z','Sp','Lp']
dp_lp_graph= ['LP Increment Membership Function', r'$\Delta$ LP (W)']


fuzzy_dp(dp_error, dp_error_name,dp_error_graph)
fuzzy_dp(dp_derrordl, dp_derrordl_name,dp_derrordl_graph)
fr_x,fr_lowest,fr_low,fr_mid,fr_high,fr_highest= fuzzy_dp(dp_fr, dp_fr_name,dp_fr_graph)
fuzzy_dp(dp_lp, dp_lp_name,dp_lp_graph)

mf = np.maximum(0.4 * fr_high, np.maximum(0.6 * fr_highest,0.4*fr_high))


xCentroid = fl.defuzz(fr_x, mf, 'centroid')

fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(fr_x, mf, linewidth=3)
plt.ylim(-1, 1)
gray = [fr_x*0.7 for x in [1, 1, 1]]

plt.vlines(xCentroid, -0.2, 1.2, color='r')
plt.text(xCentroid, -0.2, ' centroid', fontweight='bold', color='r')
xCentroid = fl.defuzz(fr_x, mf, 'centroid')


plt.show()














