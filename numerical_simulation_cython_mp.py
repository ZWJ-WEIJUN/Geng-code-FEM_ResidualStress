# -*- coding: utf-8 -*-
"""
numerical simulation of a 3*3*2 element block

@author: genli
"""
from __future__ import print_function
import math
import numpy as np
import sys
import time
import math
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from mpl_toolkits.mplot3d import Axes3D
import psutil
#from collections import Counter
#import linecache
#import os
#import tracemalloc
#import gc
#import line_profiler
#profile = line_profiler.LineProfiler()

#from memory_profiler import profile, memory_usage

#fig = plt.figure()
#ax = fig.gca(projection='3d')

import cython_material_property as cmp

"*****************************************************************************"
"""
README
symbols
cp:     heat capacity
col:    column
coef:   coefficient
fr:     feed rate
k:      conductivity
lp:     laser power
p:      parameter of equation
row:    row
T:      temperture 
time:   time 
v:      the value of parameter or variable


functions
cp:        heat capacity
k:         conductivity
film:      film coefficient
energy_T:  internal energy array
heat_flux: heat flux based on locaiton
geomety:   check the geometry definition is feasible


"""
"*****************************************************************************"
# material properties redefind in cython
#
#
#
"_____________________________________________________________________________"
# calculate the energy-temperature array
def energy_T():
# claim heat capacity temperature array   
    cp_T = [20.,   100.,  200.,  300.,  400.,  500.,  600., 700., 800., 900.,
            1000., 1100., 1200., 1300., 1385., 1450.]

# claim the values of heat capacity respective to the temperatures
    cp_v = [470e6, 490e6, 520e6, 540e6, 560e6, 570e6, 590e6, 600e6, 630e6, 
            640e6, 660e6, 670e6, 700e6, 710e6, 720e6, 830e6]

# latent heat
    latent = 2.6e11
    
    internal_energy = []
    internal_energy.append(20.*470e6)
    for i in range(0,len(cp_T)-1):
        internal_energy.append(internal_energy[-1]+(cp_T[i+1]-cp_T[i])*(cp_v[i+1]+cp_v[i])/2.)
    internal_energy[-1] += latent
    
    return internal_energy
"-----------------------------------------------------------------------------"
#
#
#
"_____________________________________________________________________________"
# define heat flux by the nodal coordinate, time, feed rate and laser power
# node is a list which contains nodal coordinates. node = [x,y,z]
def heat_flux(r,lp,p,attenuation=0.77,absorption=0.70):

# r is the distance between laser spot center and the node
#    r = math.sqrt((xc-x)**2. + (zc-z)**2.)
    
# claim the laser power distribution parameters
#    p0 = 334480.
#    p1 = -9069.4
#    p2 = -61689.
#    p3 = 16712.
#    p4 = 29465.
#    p5 = -12476.
#    p6 = -17916.
#    p = np.array([334480., -9069.4, -61689., 16712., 29465., -12476., -17916.])

# define amplify coefficient(amp_coef). Because the parameters above is obtained when the laser power
# is 1.2k watts, the heat flux should be scaled accordingly.
#    amp_coef = lp/1.2    
    
# calculate the the heat flux applied
#    q = 0.
#    if r <= 1.541:
#        q = (p[6]*r**6. + p[5]*r**5. + p[4]*r**4. + p[3]*r**3. + p[2]*r**2. + p[1]*r + p[0])*0.6667*amp_coef*attenuation*absorption
    
    return (p[6]*r**6. + p[5]*r**5. + p[4]*r**4. + p[3]*r**3. + p[2]*r**2. + p[1]*r + p[0])*0.6667*(lp/1.2)*attenuation*absorption
"-----------------------------------------------------------------------------"    
#
#
# 
"_____________________________________________________________________________"
def mod(a,b):
# calculate the module of the two numbers
    return a-math.floor(a/b)*b
"-----------------------------------------------------------------------------"
#
#
#    
"_____________________________________________________________________________"
def geometry(SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width=3.,SHM1=1.,SHM2=5.,CHM=0.7,LM=1.,WM=1.):
# check the geometry definition
# assuming the clad is built in the center of the substrate
# if the geometry is feasible, return the geometry array
# if the geometry is not feasible, quit the program 
# SubH1 is the height of the substrate top
# SubH2 is the height of the substrate bottom
# substrate bottom mesh size can be larger top mesh size since bottom temperature is not as important as top temperature     
    if mod(SubH1,SHM1)>0.001:
        print('geometry dimensions incompatible')
        print(' SubH1, SHM1 = ',SubH1, SHM1)
        sys.exit()
    elif mod(SubH2,SHM2)>0.001:
        print('geometry dimensions incompatible')
        print(' SubH2, SHM2 = ',SubH2, SHM2)
        sys.exit()
    elif mod(SubW,WM)>0.001:
        print('geometry dimensions incompatible')
        print(' SubW, WM = ',SubW, WM)
        sys.exit()
    elif mod(SubL,LM)>0.001:
        print('geometry dimensions incompatible')
        print(' SubL, LM = ',SubL, LM)
        sys.exit()
    elif mod(CladH,CHM)>0.001:
        print('geometry dimensions incompatible')
        print(' CladH, CHM = ', CladH, CHM)
        sys.exit()
    elif mod(CladW,WM)>0.001: 
        print('geometry dimensions incompatible')
        print(' CladW, WM = ', CladW, WM)
        sys.exit()
    elif mod(CladL,LM)>0.001:
        print('geometry dimensions incompatible')
        print(' CladL, LM = ', CladL, LM)
        sys.exit()
    elif mod(Width,WM)>0.001: 
        print('geometry dimensions incompatible')
        print(' Width, WM = ', Width, WM)
        sys.exit()
    elif mod(Width,LM)>0.001:  
        print('geometry dimensions incompatible')
        print(' Width, LM = ', Width, LM)
        sys.exit()
    elif mod(0.5*(SubW-CladW),WM)>0.001: 
        print('geometry dimensions incompatible')
        print(' SubW, CladW, WM = ', SubW, CladW, WM)
        sys.exit()
    elif mod(0.5*(SubL-CladL),LM)>0.001:                                        
        print('geometry dimensions incompatible')
        print(' SubL, CladL, LM = ',SubL, CladL, LM)
        sys.exit()
    else:
        return [SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width,SHM1,SHM2,CHM,LM,WM]
"-----------------------------------------------------------------------------"
# k = geometry(5.,20.,50.,50.,28.,25.,25.)
#
#
"_____________________________________________________________________________"
def thinwall_initialization(geometry_size,xd=0.,zd=0.):
# clad width must equal to thinwall width, otherwise quit the program    
    SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width,SHM1,SHM2,CHM,LM,WM = geometry_size
    if CladW != Width:
        print(quit)
        quit()
    
    SubH = SubH1 + SubH2
# number of nodes in x,y,z direction
# substrate nodes    
    x_nodes_S = int(SubL/LM+1)
    y_nodes_S = int(SubH1/SHM1+SubH2/SHM2+1)
    z_nodes_S = int(SubW/WM+1)
# clad nodes
    x_nodes_C = int(CladL/LM+1)     
    y_nodes_C = int(CladH/CHM+1)
    z_nodes_C = int(CladW/WM+1)
    
# clad corner nodes coordinates    
# clad outside corner nodes x coordinates
# if xd and zd are departure from the center of the substrate. 
# xd and zd are positive if coordinate of the clad center is larger than substrate center.    
    x0= 0.5*(SubL-CladL)+xd
    x1= 0.5*(SubL+CladL)+xd
# clad outside corner nodes z coordinates
    z0= 0.5*(SubW-CladW)+zd
    z1= 0.5*(SubW+CladW)+zd

# mesh is failed if x0 or z0 cannot be evenly divided by LM and WM respectively    
    if x0%LM>0.001 or z0%WM>0.001:
        print(quit)
        quit

# substrate nodes
    nodes_S = int(x_nodes_S*y_nodes_S*z_nodes_S)
# clad nodes
    nodes_C = int(x_nodes_C*y_nodes_C*z_nodes_C)
# interface nodes
    nodes_I = int(x_nodes_C*z_nodes_C)
# total number of nodes
    total_nodes = nodes_S + nodes_C - nodes_I

# show the basic geometry info    
    print('total_nodes = ', total_nodes)
    print('nodes_S = ', nodes_S)
    print('nodes_C = ', nodes_C)
    print('nodes_I = ', nodes_I)
    
# dynamic nodal info [Temperature, IsActive]
# create dynamic nodal array and initialize all the nodes to [25.,0]    
    node_dynamic_info = np.array([[25.,0]]*(total_nodes+1))
# initialize substrate nodes to active 
    node_dynamic_info[1:nodes_S+1] = [25.,1]  

#-----------------------------------------------------------------------------#
# static nodal info: node_connected, node_coordinate, node_ 
# connected nodal info [xp, xn, yp, yn, zp, zn]; p: positive, n: negative
# create connected nodal array and initialize to [0,0,0,0,0,0]
    node_connected = np.array([[0,0,0,0,0,0]]*(total_nodes+1))

# nodal distance is the distance between the center node and nodes connected
# create nodal distance array and initialize to [0.,0.,0.,0.,0.,0.]
    node_distance = np.array([[0.,0.,0.,0.,0.,0.]]*(total_nodes+1))    
    
# nodal coordinate info [x, y, z]
# create nodal coordinate info and initialize to [0.,0.,0.]
    node_coordinate = np.array([[0.,0.,0.]]*(total_nodes+1))
    
# nodal characteristic length info [dx, dy, dz]
# create nodal characteristci length info array
# node length is used to calculate the het transfer area
    node_length = np.array([[0.,0.,0.]]*(total_nodes+1))   
    
# nodal cross section area [axp, axn, ayp, ayn, azp, azn]    
# a stands for area, p for postive direction, n for negative direction, x,y,z are the axis
    node_cs = np.array([[0.,0.,0.,0.,0.,0.]]*(total_nodes+1))
    
# initial nodal volume     
    node_volume = np.array([0.]*(total_nodes+1))
    
# create a list of nodes arranged by layers
    nodes_in_layer = [[] for i in range(y_nodes_C)]    
    
# initialize the static nodal info with constant value
# *substrate*
# number of nodes in a layer
    layer_nodes_S = x_nodes_S*z_nodes_S
    
    for j in range(0,y_nodes_S):
        for k in range(0,z_nodes_S):
            for i in range(0,x_nodes_S):
# n: current node count, start from 0            
                n = i + k*x_nodes_S + j*x_nodes_S*z_nodes_S + 1
                
# initialize substrate nodes coordinates   
                node_coordinate[n][0] = i*LM
                if(j<=SubH2/SHM2):
                    node_coordinate[n][1] = j*SHM2
                    node_distance[n][2] = SHM2
                    node_distance[n][3] = SHM2
                else:
                    node_coordinate[n][1] = SubH2+(j-SubH2/SHM2)*SHM1
                    node_distance[n][2] = SHM1
                    node_distance[n][3] = SHM1
                node_coordinate[n][2] = k*WM
                
# initialize node distance when mesh size changes
                if j==int(SubH2/SHM2):
                    node_distance[n][2] = SHM1

# initialize the node_connected
# -1 means the node is connected to air
# if a node is connected to another node, conduction is applied
# if a node is connected to air, convection(film) is applied                  
# if a node is connected to another node, distance is equal to the mesh size
# if a node is connected to air, distance is 0.                
                if(i==0):
                    node_connected[n][1] = 0
                    node_distance[n][1]  = 0.
                else:
                    node_connected[n][1] = n-1
                    node_distance[n][1]  = LM
                
                if(i==x_nodes_S-1):
                    node_connected[n][0] = 0
                    node_distance[n][0]  = 0.
                else:
                    node_connected[n][0] = n+1
                    node_distance[n][0]  = LM
                
                if(j==0):
                    node_connected[n][3] = 0
                    node_distance[n][3]  = 0.
                else:
                    node_connected[n][3] = n-layer_nodes_S
                    
                if(j==y_nodes_S-1):
                    node_connected[n][2] = 0
                    node_distance[n][2]  = 0.
                else:
                    node_connected[n][2] = n+layer_nodes_S
                
                if(k==0):
                    node_connected[n][5] = 0
                    node_distance[n][5]  = 0.
                else:
                    node_connected[n][5] = n-x_nodes_S
                    node_distance[n][5]  = WM
                
                if(k==z_nodes_S-1):
                    node_connected[n][4] = 0
                    node_distance[n][4]  = 0.
                else:
                    node_connected[n][4] = n+x_nodes_S
                    node_distance[n][4]  = WM
                    
# initialize the node_length
                if(i==0) or (i==x_nodes_S):
                    node_length[n][0] = LM/2
                else:
                    node_length[n][0] = LM
                    
                if(j==0):
                    node_length[n][1] = SHM2/2
                elif(j<SubH2/SHM2):
                    node_length[n][1] = SHM2
                elif(j==SubH2/SHM2):
                    node_length[n][1] = (SHM1+SHM2)/2
                elif(j<y_nodes_S-1):
                    node_length[n][1] = SHM1
                elif(j==y_nodes_S-1):
                    node_length[n][1] = SHM1/2
                    
                if(k==0) or (k==z_nodes_S):
                    node_length[n][2] = WM/2
                else:
                    node_length[n][2] = WM
                    
# *interlayer*
# number of nodes in a clad layer 
    layer_nodes_C = int(x_nodes_C*z_nodes_C)

# the top left node count in the substrate-clad interface    
# the first line: the number of total nodes of substrate except top surface
# the second line: the number of nodes before the first interface node    

    n0_interface = int(x_nodes_S*z_nodes_S*(y_nodes_S-1) + \
                       z0/WM*x_nodes_S + x0/LM + 1)
    print('x_nodes_S = ',x_nodes_S)
    print('y_nodes_S = ',y_nodes_S)
    print('z_nodes_S = ',z_nodes_S)
    print('z0 = ',z0)
    print('WM = ',WM)
    print('x0 = ',x0)
    print('LM = ',LM)
    print('n0_interface = ',n0_interface)
# re-initialize the interface nodes
# node_coordinate and node_lenth are the same, only node_connected needs to be re-initialize
# use interface_node_counter to keep track of the interface nodes                   
    interface_node_counter = 0    
# create an array to hold all the interface nodes
    interface_nodes = np.empty([0,],dtype=int)   

    for k in range(0,z_nodes_C):
        for i in range(0,x_nodes_C):
# current node in substrate top surface layer, start from the first node in interface
            current_node = int(n0_interface + i + k*x_nodes_S)
# determine whether the current node is the interface node
            if node_coordinate[current_node][0] >= x0 and \
               node_coordinate[current_node][0] <= x1 and \
               node_coordinate[current_node][2] >= z0 and \
               node_coordinate[current_node][2] <= z1:
# put all the interface nodes into interface_nodes array            
                   interface_nodes = np.append(interface_nodes,current_node)
# node_above is the node above the current interface node 
                   node_above = nodes_S + interface_node_counter
                   node_connected[current_node][2] = node_above
                   node_connected[node_above][3] = current_node
                   interface_node_counter += 1
# initialize the node distance of interface node
                   node_distance[current_node][2] = CHM
                    
# interface_node_counter should equal to a layer of clad node after the loop above
# this is used to test whether the code above is working properly                   
    if(interface_node_counter != layer_nodes_C):
        print('layer_nodes_C = ', layer_nodes_C)
        print('interface_node_counter = ', interface_node_counter)
        print('the loop above is NOT working properly')
    else:
        print('the loop is working properly')
        

#*clad*#        
# initialize the clad except the interface layer
# clad_node_counter is used to keep track of the clad node
    nodes_in_layer[0] = interface_nodes           
    clad_node_counter = 0        
    for j in range(1,y_nodes_C):
        for k in range(0,z_nodes_C):
            for i in range(0,x_nodes_C):          
# the coordinate of point according to the loop iterator
# the point is not necessarily the clad node               
                x = x0 + i*LM
                y = SubH + j*CHM
                z = z0 + k*WM
# whether the point is a clad node or not is determined by its coordinates                 
                if x>=x0 and x<=x1 and\
                   z>=z0 and z<=z1:                     
# n is the current node
                    n = nodes_S + clad_node_counter + 1
                    
# initialize the layer_nodes by layer index
                    nodes_in_layer[j].append(n)
# test
                    if(n>=(total_nodes+1)):
                        print('clad_node_counter', clad_node_counter)
                        print('i,j,k = ',i,j,k)
                        print('x,y,z = ',x,y,z)
                        print('n = ', n)
# initialize coordinates                    
                    node_coordinate[n] = np.array([x,y,z])
# initialize connected nodes and node distance
                    if abs(x-x0)<0.001:
                        node_connected[n][1] = 0
                        node_distance[n][1]  = 0.
                    else:
                        node_connected[n][1] = n-1
                        node_distance[n][1]  = LM
                    
                    if abs(x-x1)<0.001:
                        node_connected[n][0] = 0
                        node_distance[n][0]  = 0.
                    else:
                        node_connected[n][0] = n+1
                        node_distance[n][0]  = LM
                        
                    if y>(SubH+CHM):
                        node_connected[n][3] = n - layer_nodes_C
                    if y>=(SubH+CHM):    
                        node_distance[n][3]  = CHM
                    
                    if abs(y-(SubH+CladH))<0.001:
                        node_connected[n][2] = 0
                        node_distance[n][2]  = 0.
                    else:
                        node_connected[n][2] = n + layer_nodes_C
                        node_distance[n][2]  = CHM
                    
                    if abs(z-z0)<0.001:
                        node_connected[n][5] = 0
                        node_distance[n][5]  = 0.
                    else:
                        node_connected[n][5] = n - x_nodes_C
                        node_distance[n][5]  = WM
                        
                    if abs(z-z1)<0.001:
                        node_connected[n][4] = 0
                        node_distance[n][4]  = 0.
                    else:
                        node_connected[n][4] = n + x_nodes_C
                        node_distance[n][4]  = WM
                        
# initialize the clad node_length                
                    if abs(x-x0)<0.001 or abs(x-x1)<0.001:
                        node_length[n][0] = LM/2
                    else:
                        node_length[n][0] = LM
                    
                    node_length[n][1] = CHM/2
                    
                    if abs(z-z0)<0.001 or abs(z-z1)<0.001:
                        node_length[n][2] = WM/2
                    else:
                        node_length[n][2] = WM
                    
                    clad_node_counter += 1
                    
                    
# initialize node volume
    for n in range(total_nodes+1):
        node_volume[n] = np.prod(node_length[n])
        node_cs[n][0] = node_length[n][1]*node_length[n][2]
        node_cs[n][1] = node_length[n][1]*node_length[n][2]
        node_cs[n][2] = node_length[n][0]*node_length[n][2]
        node_cs[n][3] = node_length[n][0]*node_length[n][2]
        node_cs[n][4] = node_length[n][0]*node_length[n][1]
        node_cs[n][5] = node_length[n][0]*node_length[n][1]                     
    
    
    print('len(interface_nodes) = ', len(interface_nodes))
    return node_dynamic_info, node_connected, node_distance, node_coordinate, node_length, node_volume, node_cs, nodes_in_layer                


    
    
    
    
"-----------------------------------------------------------------------------"
#
#
#    
"_____________________________________________________________________________"
def block_initialization(geometry_size):
# symbols in the x,y,z direction (x,y,z)=(i,j,k)=(L,H,W)
# nodes are initialized from bottom to top, in order of SubH2, SUbH1, ClADH    
    SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width,SHM1,SHM2,CHM,LM,WM = geometry_size 
    SubH = SubH1 + SubH2
# number of nodes in x,y,z direction
# substrate nodes    
    x_nodes_S = int(SubL/LM+1)
    y_nodes_S = int(SubH1/SHM1+SubH2/SHM2+1)
    z_nodes_S = int(SubW/WM+1)
# clad nodes
    x_nodes_C = int(CladL/LM+1)     
    y_nodes_C = int(CladH/CHM+1)
    z_nodes_C = int(CladW/WM+1)
    
# clad corner nodes coordinates    
# clad outside corner nodes x coordinates
    xo_C1 = 0.5*(SubL-CladL)
    xo_C2 = 0.5*(SubL+CladL)
# clad outside corner nodes z coordinates
    zo_C1 = 0.5*(SubW-CladW)
    zo_C2 = 0.5*(SubW+CladW)
# clad inside corner nodes x coordinates
    xi_C1 = xo_C1 + Width
    xi_C2 = xo_C2 - Width
# clad inside corner nodes z coordinates
    zi_C1 = zo_C1 + Width
    zi_C2 = zo_C2 - Width 
    
# substrate nodes
    nodes_S = int(x_nodes_S*y_nodes_S*z_nodes_S)
# clad nodes
    nodes_C = int((x_nodes_C*z_nodes_C - (x_nodes_C-2*Width/LM-2)*(z_nodes_C-2*Width/WM-2))*y_nodes_C)
# interface nodes
    nodes_I = int(x_nodes_C*z_nodes_C - (x_nodes_C-2*Width/LM-2)*(z_nodes_C-2*Width/WM-2))
# total number of nodes
    total_nodes = nodes_S + nodes_C - nodes_I

# show the basic geometry info    
    print('total_nodes = ', total_nodes)
    print('nodes_S = ', nodes_S)
    print('nodes_C = ', nodes_C)
    print('nodes_I = ', nodes_I)
    
# dynamic nodal info [Temperature, IsActive]
# create dynamic nodal array and initialize all the nodes to [25.,0]    
    node_dynamic_info = np.array([[25.,0]]*(total_nodes+1))
# initialize substrate nodes to active 
    node_dynamic_info[1:nodes_S+1] = [25.,1]  

#-----------------------------------------------------------------------------#
# static nodal info: node_connected, node_coordinate, node_ 
# connected nodal info [xp, xn, yp, yn, zp, zn]; p: positive, n: negative
# create connected nodal array and initialize to [0,0,0,0,0,0]
    node_connected = np.array([[0,0,0,0,0,0]]*(total_nodes+1))

# nodal distance is the distance between the center node and nodes connected
# create nodal distance array and initialize to [0.,0.,0.,0.,0.,0.]
    node_distance = np.array([[0.,0.,0.,0.,0.,0.]]*(total_nodes+1))    
    
# nodal coordinate info [x, y, z]
# create nodal coordinate info and initialize to [0.,0.,0.]
    node_coordinate = np.array([[0.,0.,0.]]*(total_nodes+1))
    
# nodal characteristic length info [dx, dy, dz]
# create nodal characteristci length info array
# node length is used to calculate the het transfer area
    node_length = np.array([[0.,0.,0.,0.,0.,0.]]*(total_nodes+1))
    
# nodal cross section area [axp, axn, ayp, ayn, azp, azn]    
# a stands for area, p for postive direction, n for negative direction, x,y,z are the axis
    node_cs = np.array([[0.,0.,0.,0.,0.,0.]]*(total_nodes+1))
    
# initial nodal volume     
    node_volume = np.array([0.]*(total_nodes+1))
    
# create a list of nodes arranged by layers
    nodes_in_layer = [[] for i in range(y_nodes_C)]
    
# create a list of inner corner nodes
    inner_corner_nodes = []    
    
# initialize the static nodal info with constant value
# *substrate*
# number of nodes in a layer
    layer_nodes_S = x_nodes_S*z_nodes_S
    
    for j in range(0,y_nodes_S):
        for k in range(0,z_nodes_S):
            for i in range(0,x_nodes_S):
# n: current node count, start from 0            
                n = i + k*x_nodes_S + j*x_nodes_S*z_nodes_S + 1
                
# initialize substrate nodes coordinates   
                node_coordinate[n][0] = i*LM
                if(j<=SubH2/SHM2):
                    node_coordinate[n][1] = j*SHM2
                    node_distance[n][2] = SHM2
                    node_distance[n][3] = SHM2
                else:
                    node_coordinate[n][1] = SubH2+(j-SubH2/SHM2)*SHM1
                    node_distance[n][2] = SHM1
                    node_distance[n][3] = SHM1
                node_coordinate[n][2] = k*WM
                
# initialize node distance when mesh size changes
                if j==int(SubH2/SHM2):
                    node_distance[n][2] = SHM1

# initialize the node_connected
# -1 means the node is connected to air
# if a node is connected to another node, conduction is applied
# if a node is connected to air, convection(film) is applied                  
# if a node is connected to another node, distance is equal to the mesh size
# if a node is connected to air, distance is 0.                
                if(i==0):
                    node_connected[n][1] = 0
                    node_distance[n][1]  = 0.
                else:
                    node_connected[n][1] = n-1
                    node_distance[n][1]  = LM
                
                if(i==x_nodes_S-1):
                    node_connected[n][0] = 0
                    node_distance[n][0]  = 0.
                else:
                    node_connected[n][0] = n+1
                    node_distance[n][0]  = LM
                
                if(j==0):
                    node_connected[n][3] = 0
                    node_distance[n][3]  = 0.
                else:
                    node_connected[n][3] = n-layer_nodes_S
                    
                if(j==y_nodes_S-1):
                    node_connected[n][2] = 0
                    node_distance[n][2]  = 0.
                else:
                    node_connected[n][2] = n+layer_nodes_S
                
                if(k==0):
                    node_connected[n][5] = 0
                    node_distance[n][5]  = 0.
                else:
                    node_connected[n][5] = n-x_nodes_S
                    node_distance[n][5]  = WM
                
                if(k==z_nodes_S-1):
                    node_connected[n][4] = 0
                    node_distance[n][4]  = 0.
                else:
                    node_connected[n][4] = n+x_nodes_S
                    node_distance[n][4]  = WM
                    
# initialize the node_length
                if(i==0) or (i==x_nodes_S):
                    node_length[n][0] = LM/2
                    node_length[n][1] = LM/2
                else:
                    node_length[n][0] = LM
                    node_length[n][1] = LM
                    
                if(j==0):
                    node_length[n][2] = SHM2/2
                    node_length[n][3] = SHM2/2
                elif(j<SubH2/SHM2):
                    node_length[n][2] = SHM2
                    node_length[n][3] = SHM2
                elif(j==SubH2/SHM2):
                    node_length[n][2] = (SHM1+SHM2)/2
                    node_length[n][3] = (SHM1+SHM2)/2
                elif(j<y_nodes_S-1):
                    node_length[n][2] = SHM1
                    node_length[n][3] = SHM1
                elif(j==y_nodes_S-1):
                    node_length[n][2] = SHM1/2
                    node_length[n][3] = SHM1/2
                    
                if(k==0) or (k==z_nodes_S):
                    node_length[n][4] = WM/2
                    node_length[n][5] = WM/2
                else:
                    node_length[n][4] = WM
                    node_length[n][5] = WM
                
# *clad*
# number of nodes in a clad layer 
    layer_nodes_C = int(x_nodes_C*z_nodes_C - (x_nodes_C-2*Width/LM-2)*(z_nodes_C-2*Width/LM-2))

# the top left node count in the substrate-clad interface    
# the first line: the number of total nodes of substrate except top surface
# the second line: the number of nodes before the first interface node    

    n0_interface = int(x_nodes_S*z_nodes_S*(y_nodes_S-1) + \
                       zo_C1/WM*x_nodes_S + xo_C1/LM + 1)

# re-initialize the interface nodes
# node_coordinate and node_lenth are the same, only node_connected needs to be re-initialize
# use interface_node_counter to keep track of the interface nodes                   
    interface_node_counter = 0    
# create an array to hold all the interface nodes
#    interface_nodes = np.empty([0,],dtype=int)
    interface_nodes = []    
    for k in range(0,z_nodes_C):
        for i in range(0,x_nodes_C):
# current node in substrate top surface layer, start from the first node in interface
            current_node = int(n0_interface + i + k*x_nodes_S)
# determine whether the current node is the interface node
            if node_coordinate[current_node][0] >= xo_C1 and \
               node_coordinate[current_node][0] <= xo_C2 and \
               node_coordinate[current_node][2] >= zo_C1 and \
               node_coordinate[current_node][2] <= zo_C2 and not \
               (node_coordinate[current_node][0] > xi_C1 and \
                node_coordinate[current_node][0] < xi_C2 and \
                node_coordinate[current_node][2] > zi_C1 and \
                node_coordinate[current_node][2] < zi_C2):
# put all the interface nodes into interface_nodes array            
#                   interface_nodes = np.append(interface_nodes,current_node)
                   interface_nodes.append(current_node)
# node_above is the node above the current interface node 
                   node_above = (nodes_S+1) + interface_node_counter
                   node_connected[current_node][2] = node_above
                   node_connected[node_above][3] = current_node
                   interface_node_counter += 1
# initialize the node distance of interface node
                   node_distance[current_node][2] = CHM
                   
                    
# interface_node_counter should equal to a layer of clad node after the loop above
# this is used to test whether the code above is working properly                   
    if(interface_node_counter != layer_nodes_C):
        print('layer_nodes_C = ', layer_nodes_C)
        print('interface_node_counter = ', interface_node_counter)
        print('the loop above is NOT working properly')
    else:
        print('the loop is working properly')
               
# initialize the clad except the interface layer
# clad_node_counter is used to keep track of the clad node
# initialize layer_nodes
    nodes_in_layer[0] = interface_nodes        
    clad_node_counter = 0        
    for j in range(1,y_nodes_C):
# inner corner nodes
        inner_nodes = [0.]*4        
        
        for k in range(0,z_nodes_C):
            for i in range(0,x_nodes_C):          
# the coordinate of point according to the loop iterator
# the point is not necessarily the clad node               
                x = xo_C1 + i*LM
                y = SubH  + j*CHM
                z = zo_C1 + k*WM
                
# the point is whether a clad or not is determined by its coordinates                 
                if x>=xo_C1 and x<=xo_C2 and\
                   z>=zo_C1 and z<=zo_C2 and not\
                   (x>xi_C1 and x<xi_C2  and\
                    z>zi_C1 and z<zi_C2):                       
# n is the current node
                    n = nodes_S + clad_node_counter + 1
# initialize the layer_nodes by layer index
                    nodes_in_layer[j].append(n)

# find inner corner nodes                    
                    if x==xi_C1 and z==zi_C1:
                        inner_nodes[0] = n
                    elif x==xi_C2 and z==zi_C1:
                        inner_nodes[1] = n
                    elif x==xi_C2 and z==zi_C2:
                        inner_nodes[2] = n
                    elif x==xi_C1 and z==zi_C2:
                        inner_nodes[3] = n
# test
                    if(n>=(total_nodes+1+1)):
                        print('clad_node_counter', clad_node_counter)
                        print('i,j,k = ',i,j,k)
                        print('x,y,z = ',x,y,z)
                        print('n = ', n)
# initialize coordinates                    
                    node_coordinate[n] = np.array([x,y,z])
# initialize connected nodes
                    if x==xo_C1 or (x==xi_C2 and z>zi_C1 and z<zi_C2):
                        node_connected[n][1] = 0
                        node_distance[n][1]  = 0.
                    else:
                        node_connected[n][1] = n-1
                        node_distance[n][1]  = LM
                    
                    if x==xo_C2 or (x==xi_C1 and z>zi_C1 and z<zi_C2):
                        node_connected[n][0] = 0
                        node_distance[n][0]  = 0.
                    else:
                        node_connected[n][0] = n+1
                        node_distance[n][0]  = LM
                        
                    if y>(SubH+CHM):
                        node_connected[n][3] = n - layer_nodes_C
                    if y>=(SubH+CHM):    
                        node_distance[n][3]  = CHM
                    
                    if y==(SubH+CladH):
                        node_connected[n][2] = 0
                        node_distance[n][2]  = 0.
                    else:
                        node_connected[n][2] = n + layer_nodes_C
                        node_distance[n][2]  = CHM
                    
                    if z==zo_C1:
                        node_connected[n][4] = n + x_nodes_C
                        node_connected[n][5] = 0
                        node_distance[n][4]  = WM
                        node_distance[n][5]  = 0.
                    elif z<zi_C1:
                        node_connected[n][4] = n + x_nodes_C
                        node_connected[n][5] = n - x_nodes_C
                        node_distance[n][4]  = WM
                        node_distance[n][5]  = WM
                    elif z==zi_C1:
                        node_connected[n][5] = n - x_nodes_C
                        node_distance[n][5]  = WM
                        if x>xi_C1 and x<xi_C2:
                            node_connected[n][4] = 0
                            node_distance[n][4]  = 0.
                        elif x<=xi_C1:
                            node_connected[n][4] = n + x_nodes_C
                            node_distance[n][4]  = WM
                        else:
                            node_connected[n][4] = n + int((Width/WM+1)*2)
                            node_distance[n][4]  = WM
                    elif z==zi_C1+WM:
                        node_connected[n][4] = n + int((Width/WM+1)*2)
                        node_distance[n][4]  = WM
                        if x<=xi_C1:
                            node_connected[n][5] = n - x_nodes_C
                            node_distance[n][5]  = WM
                        else:
                            node_connected[n][5] = n - int((Width/WM+1)*2)
                            node_distance[n][5]  = WM
                    elif z>(zi_C1+WM) and z<(zi_C2-WM):
                        node_connected[n][4] = n + int((Width/WM+1)*2)
                        node_connected[n][5] = n - int((Width/WM+1)*2)
                        node_distance[n][4]  = WM
                        node_distance[n][5]  = WM
                    elif z==zi_C2-WM:
                        node_connected[n][5] = n - int((Width/WM+1)*2)
                        node_distance[n][5]  = WM
                        if x<=xi_C1:
                            node_connected[n][4] = n + int((Width/WM+1)*2)
                            node_distance[n][4]  = WM
                        else:
                            node_connected[n][4] = n + x_nodes_C
                            node_distance[n][4]  = WM
                    elif z==zi_C2:
                        node_connected[n][4] = n + x_nodes_C
                        node_distance[n][4]  = WM
                        if x<=xi_C1:
                            node_connected[n][5] = n - int((Width/WM+1)*2)
                            node_distance[n][5]  = WM
                        elif x>xi_C1 and x<xi_C2:
                            node_connected[n][5] = 0
                            node_distance[n][5]  = 0.
                        else:
                            node_connected[n][5] = n - x_nodes_C
                            node_distance[n][5]  = WM
                    elif z>zi_C2 and z<zo_C2:
                        node_connected[n][4] = n + x_nodes_C
                        node_connected[n][5] = n - x_nodes_C
                        node_distance[n][4]  = WM
                        node_distance[n][5]  = WM
                    elif z==zo_C2:
                        node_connected[n][4] = 0
                        node_connected[n][5] = n - x_nodes_C
                        node_distance[n][4]  = 0.
                        node_distance[n][5]  = WM
                        
# initialize the clad node_length                
                    if x==xo_C1 or x==xo_C2 or \
                      (x==xi_C1 and z>=zi_C1 and z<=zi_C2) or \
                      (x==xi_C2 and z>=zi_C1 and z<=zi_C2):
                        node_length[n][0] = LM/2
                        node_length[n][1] = LM/2
                    else:
                        node_length[n][0] = LM
                        node_length[n][1] = LM
                    
                    node_length[n][2] = CHM
                    node_length[n][3] = CHM
                    
                    if z==zo_C1 or z==zo_C2 or \
                      (z==zi_C1 and x>=xi_C1 and x<=xi_C2) or \
                      (z==zi_C2 and x>=xi_C1 and x<=xi_C2):
                        node_length[n][4] = WM/2
                        node_length[n][5] = WM/2
                    else:
                        node_length[n][4] = WM
                        node_length[n][5] = WM
                    
                    clad_node_counter += 1
        inner_corner_nodes.append(inner_nodes)

# initialize node volume and cross section area
    for n in range(total_nodes+1):
        node_volume[n] = node_length[n][0]*node_length[n][2]*node_length[n][4]   
        node_cs[n][0] = node_length[n][2]*node_length[n][4]
        node_cs[n][1] = node_length[n][2]*node_length[n][4]
        node_cs[n][2] = node_length[n][0]*node_length[n][4]
        node_cs[n][3] = node_length[n][0]*node_length[n][4]
        node_cs[n][4] = node_length[n][0]*node_length[n][2]
        node_cs[n][5] = node_length[n][0]*node_length[n][2]
        
#    for n in range(nodes_S+1,total_nodes+1):
#        node_volume[n] = node_volume[n]*2

# interface nodes have different upper half volume/cs and lower half volume/cs
    horizontal_directions = [0,1,4,5]
    for n in interface_nodes:
        n_above = node_connected[n][2]
        node_cs[n][2] =  node_length[n_above][1]*node_length[n_above][5]
        for d in horizontal_directions:
            n_conn     = node_connected[n][d]
            conn_above = node_connected[n_conn][2]
            if conn_above:    
                node_cs[n][d] += node_length[n_above][3]*node_length[n_above][5-d]
#        node_cs[n][1] += node_length[n_above][3]*node_length[n_above][5]/2.
#        node_cs[n][4] += node_length[n_above][1]*node_length[n_above][3]/2.
#        node_cs[n][5] += node_length[n_above][1]*node_length[n_above][3]/2.

# clad inner corner nodal cross section area need to be re-defined
# the nodal cross section area of many clad nodes are depending on the activation situation 
# of their connected nodes.        
    for k in range(y_nodes_C-1):
# corner nodes in 4 directions
        n1 = inner_corner_nodes[k][0]
        node_cs[n1][0] = WM*CHM/2.
        node_cs[n1][1] = WM*CHM
        node_cs[n1][2] = LM*WM*3./4.
        node_cs[n1][3] = LM*WM*3./4.
        node_cs[n1][4] = WM*CHM/2.
        node_cs[n1][5] = WM*CHM
        node_length[n1][0] = WM/2.
        node_length[n1][1] = WM
        node_volume[n1] = node_cs[n1][2]*CHM

        n2 = inner_corner_nodes[k][1]
        node_cs[n2][0] = WM*CHM
        node_cs[n2][1] = WM*CHM/2.
        node_cs[n2][2] = LM*WM*3./4.
        node_cs[n2][3] = LM*WM*3./4.
        node_cs[n2][4] = WM*CHM/2.
        node_cs[n2][5] = WM*CHM
        node_length[n2][4] = LM/2.
        node_length[n2][5] = LM
        node_volume[n2] = node_cs[n2][2]*CHM        
        
        n3 = inner_corner_nodes[k][2]
        node_cs[n3][0] = WM*CHM
        node_cs[n3][1] = WM*CHM/2.
        node_cs[n3][2] = LM*WM*3./4.
        node_cs[n3][3] = LM*WM*3./4.
        node_cs[n3][4] = WM*CHM
        node_cs[n3][5] = WM*CHM/2.  
        node_length[n3][4] = WM
        node_length[n3][5] = WM/2.  
        node_volume[n3] = node_cs[n3][2]*CHM        
        
        n4 = inner_corner_nodes[k][3]
        node_cs[n4][0] = WM*CHM/2.
        node_cs[n4][1] = WM*CHM
        node_cs[n4][2] = LM*WM*3./4.
        node_cs[n4][3] = LM*WM*3./4.
        node_cs[n4][4] = WM*CHM
        node_cs[n4][5] = WM*CHM/2.                            
        node_length[n4][0] = LM/2.
        node_length[n4][1] = LM
        node_volume[n4] = node_cs[n4][2]*CHM

# redefine the cross section area of the interface nodes whose nodes above are inner corner nodes                    
    n1_below = node_connected[inner_corner_nodes[0][0]][3]
    node_cs[n1_below][2] = LM*WM*3./4.
    node_cs[n1_below][0] = WM*node_distance[n1_below][3]/2.
    node_cs[n1_below][1] = WM*node_distance[n1_below][3]/2. 
    node_cs[n1_below][4] = LM*node_distance[n1_below][3]/2. 
    node_cs[n1_below][5] = LM*node_distance[n1_below][3]/2. 
    
    n2_below = node_connected[inner_corner_nodes[0][1]][3]
    node_cs[n2_below][2] = LM*WM*3./4.
    node_cs[n2_below][0] = WM*node_distance[n2_below][3]/2. 
    node_cs[n2_below][1] = WM*node_distance[n2_below][3]/2. 
    node_cs[n2_below][4] = LM*node_distance[n2_below][3]/2. 
    node_cs[n2_below][5] = LM*node_distance[n2_below][3]/2. 
    
    n3_below = node_connected[inner_corner_nodes[0][2]][3]
    node_cs[n3_below][2] = LM*WM*3./4.
    node_cs[n3_below][0] = WM*node_distance[n3_below][3]/2.
    node_cs[n3_below][1] = WM*node_distance[n3_below][3]/2. 
    node_cs[n3_below][4] = LM*node_distance[n3_below][3]/2. 
    node_cs[n3_below][5] = LM*node_distance[n3_below][3]/2.    

    n4_below = node_connected[inner_corner_nodes[0][3]][3]
    node_cs[n4_below][2] = LM*WM*3./4.
    node_cs[n4_below][0] = WM*node_distance[n4_below][3]/2.
    node_cs[n4_below][1] = WM*node_distance[n4_below][3]/2. 
    node_cs[n4_below][4] = LM*node_distance[n4_below][3]/2. 
    node_cs[n4_below][5] = LM*node_distance[n4_below][3]/2.     
        
    
#    print('len(interface_nodes) = ', len(interface_nodes))
    return node_dynamic_info, node_connected, node_distance, node_coordinate, node_length, node_volume, node_cs, nodes_in_layer
"-----------------------------------------------------------------------------"
#
#
#    
"_____________________________________________________________________________"
   
    
    
"-----------------------------------------------------------------------------"
#    
#   
#      
"""
"_____________________________________________________________________________"
# An ulternative method to construct the model
def nodal_info_initialization(geometry_size):
# symbols in the x,y,z direction (x,y,z)=(i,j,k)=(L,H,W)
# nodes are initialized from bottom to top, in order of SubH2, SUbH1, ClADH    

    SubH1,SubH2,SubW,SubL,CladH,CladW,CladL,Width,SHM1,SHM2,CHM,WM,LM = geometry_size     
# nodal info [x,y,z,Temperature,active]
    node = []    
# initialize nodes in substrate SubH2, SubH1
    for j in range (0,SubH2/SHM2):
        for k in range (0,SubW/WM+1):
            for i in range (0,SubL/LM+1):
                node.append([i*LM,j*SHM2,k*WM,25.,1])
    
    for j in range (0,SubH1/SHM1+1):
        for k in range (0,SubW/WM+1):
            for i in range (0,SubL/LM+1):
                node.append([i*LM,j*SHM1+SubH2,k*WM,25.,1])
                
# initialize nodes in clad
    OutsideCorner1x = 0.5*(SubL-CladL)
    OutsideCorner1z = 0.5*(SubW-CladW)
    OutsideCorner2x = 0.5*(SubL+CladL)
    OutsideCorner2z = 0.5*(SubW-CladW)
    
    InsideCorner1x = OutsideCorner1x+Width
    
    for j in range (1,CladH/CHM+1): 
        for k in range (0, CladW/WM+1):
            for i in range (0, CladL/LM+1):
                node.append([])

"-----------------------------------------------------------------------------"
"""
#
#
#
"_____________________________________________________________________________"
# calculate the nodal temperature for next increment
# n_T current nodal temperature; 
def T_next(rou_energy,ener_i,cp_T,cp_v,range_Tnext):
# claim heat capacity temperature array   
#    cp_T = [20.,   100.,  200.,  300.,  400.,  500.,  600., 700., 800., 900.,
#            1000., 1100., 1200., 1300., 1385., 1450.]

# claim the values of heat capacity respective to the temperatures
#    cp_v = [470e6, 490e6, 520e6, 540e6, 560e6, 570e6, 590e6, 600e6, 630e6, 
#            640e6, 660e6, 670e6, 700e6, 710e6, 720e6, 830e6]

# internal_energy density list: ener_i
#    energy = energy_T()
# density rou    
#    rou = 7.95e-9
# nodal volume V   
    
# calculate the energy density for current temperature 
# rou_energy: energy per mass    
#    for i in range(0,len(ener_i)-1):
#        if(n_T<=cp_T[i+1]):
#            rou_energy = ener_i[i] + (n_T-cp_T[i])/(cp_T[i+1]-cp_T[i])*(ener_i[i+1]-ener_i[i]) 
#            break
#    if(n_T>cp_T[-1]):
#        rou_energy = ener_i[-1] + (n_T-cp_T[-1])*cp_v[-1]

# calculate the energy density for next increment
#    rou_energy += input_energy/(rou*V)
        
# calculate the temperature  
    for i in range_Tnext:
        if(rou_energy<ener_i[i+1]):
            #Temperature_next = cp_T[i] + (rou_energy-ener_i[i])/(ener_i[i+1]-ener_i[i])*(cp_T[i+1]-cp_T[i])
            #return Temperature_next
            return cp_T[i] + (rou_energy-ener_i[i])/(ener_i[i+1]-ener_i[i])*(cp_T[i+1]-cp_T[i])
    if(rou_energy>=ener_i[-1]):
        #Temperature_next = cp_T[-1] + (rou_energy-ener_i[-1])/cp_v[-1]
        return cp_T[-1] + (rou_energy-ener_i[-1])/cp_v[-1]
#    return Temperature_next       
    
"-----------------------------------------------------------------------------"
#
#
#
"_____________________________________________________________________________"
def zigzag(xc,zc,layer,direction,dwell_tc,dwell_t,xc0,xc1,zc0,zc1,fr,t_i,cladding_direction='x'):    
   
#-------------------------parameters------------------------------------------#
# xc: current x coordinate of laser spot center; zc: current z coordinate of laser spot center     
# layer: the deposition layer
# direction: the laser spot moves back and forth in zig-zag mod; 
#            the laser moving direction is eihter 1 or -1,
#            if laser spot is moving in positive direction, direction is 1.
#            if laser sopt is moving in negative direction, direction is -1.
    
# dwell_tc: the current dwell time left. 
#           if the dwell_tc is greater than 0, the laser spot center is moved to 
#           a location far from the actual laser path. Therefore, there is no laser power
#           during dwell time.   
#           if dwell_tc is smaller than 0, the dwell time is over.  
# dwell_t: dwell time; dwell time is required for thin wall deposition.     

# xc0------------xc1 :clad length/width in x direction
# zc0------------zc1 :clad length/width in z direction    
# xc0: the initial x coordinate of the positive direction
# zc0: the initial z coordinate of the positive direction
# xc1: the initial x coordinate of the negative direction
# zc1: the initial z coordinate of the negative direction    

# fr: feed rate
# t_i: time incremen
# cladding direction: cladding either in 'x' direction or 'z' direction; 
#                     default option is 'x'    
#-----------------------------------------------------------------------------#
    
    
# x_far, z_far: a set of coordinate very far from the actual laser path,
# so that laser is off during dwell time.     
    x_far = -100.
    z_far = -100.

# xc_next: x coordinate of laser spot center in next increment 
# zc_next: x coordinate of laser spot center in next increment
    
    if dwell_tc>0.:
        xc_next = x_far
        zc_next = z_far
        dwell_tc -= t_i
        if dwell_tc>0.:
            return xc_next, zc_next, layer, direction, dwell_tc 
        elif dwell_tc<=0.:
            direction = direction*-1
            if cladding_direction=='x':
                if direction==1:
                    xc_next = xc0-dwell_tc*fr
                else:
                    xc_next = xc1+dwell_tc*fr
                zc_next = zc0
            elif cladding_direction=='z':
                if direction==1:
                    zc_next = zc0-dwell_tc*fr
                else:
                    zc_next = zc1+dwell_tc*fr
                xc_next = xc0
            dwell_tc = 0.
            layer += 1
            
            return xc_next, zc_next, layer, direction, dwell_tc 

# the deposition is on x-z plane
# if the cladding direction is along x axis
    if cladding_direction=='x':
        if direction==1:
            xc_next = xc+t_i*fr
            if xc_next>xc1:
                if dwell_t:
                    dwell_tc = dwell_t-(xc_next-xc1)/fr
                else:
                    xc_next = xc1-(xc_next-xc1)
                    direction = -1
                    layer += 1
        else:
            xc_next = xc-t_i*fr
            if xc_next<xc0:
                if dwell_t:
                    dwell_tc = dwell_t-(xc0-xc_next)/fr
                else:
                    xc_next = xc0+(xc0-xc_next)
                    direction = 1
                    layer += 1
        zc_next = zc0
# if the cladding direction is along z axis       
    elif cladding_direction=='z':
        if direction==1:
            zc_next = zc+t_i*fr
            if zc_next>zc1:
                if dwell_t:
                    dwell_tc = dwell_t-(zc_next-zc1)/fr
                else:
                    zc_next = zc1-(zc_next-zc1)
                    direction = -1
                    layer += 1
        else:
            zc_next = zc-t_i*fr
            if zc_next<zc0:
                if dwell_t:
                    dwell_tc = dwell_t-(zc0-zc_next)/fr
                else:
                    zc_next = zc0+(zc0-zc_next)
                    direction = 1
                    layer += 1
        xc_next = xc0
        
    return xc_next, zc_next, layer, direction, dwell_tc
"-----------------------------------------------------------------------------"
#
#
#
"_____________________________________________________________________________"
def clockwise_square(xc,zc,layer,xc0,xc1,zc0,zc1,fr,t_i,cladding_direction='clockwise'):
    
#-----------------parameters--------------------------------------------------#    
# xc,zc are the current laser center coordinate
# xc0, zc0 are the initial laser center coordinate
# xc1, zc1 are the laser center coordinate of oppsite corner of initial laser center 
# fr: feed rate
# t_i: time increment
# layer: current layer
#-----------------------------------------------------------------------------#    
# assuming the cladding direction is clockwise    
    if zc==zc0:
        xc_next = xc + fr*t_i
        zc_next = zc0
        if xc_next>=xc1:
            zc_next = zc0 + (xc_next-xc1)
            xc_next = xc1
    elif xc==xc1:
        xc_next = xc1
        zc_next = zc + fr*t_i
        if zc_next>=zc1:
            xc_next = xc1 - (zc_next-zc1)
            zc_next = zc1
    elif zc==zc1:
        xc_next = xc - fr*t_i
        zc_next = zc1
        if xc_next<=xc0:
            zc_next = zc1 - (xc0-xc_next)
            xc_next = xc0
    elif xc==xc0:
        xc_next = xc0
        zc_next = zc - fr*t_i
        if zc_next<=zc0:
            xc_next = xc0 + (zc0-zc_next)
            zc_next = zc0
            layer += 1
    
    return xc_next, zc_next, layer
"-----------------------------------------------------------------------------"
#
#
#
"_____________________________________________________________________________"
def fuzzy_control(lp_current, fr_current, median_temp, layer, is_fr_control,
                  obj_temp, activation_threshold, stable_threshold,
                  dp_error,dp_derrordl,dp_fr,dp_lp,
                  dp_error_ub=50.,dp_error_lb=-50.,dp_dedl_ub=30.,dp_dedl_lb=-30.):
# fuzzy control unit: use fuzzy logic to control the feed rate based on the information
# return feed rate for next layer
    
# fuzzy control is composed of 3 modules: fuzzification, inference and defuzzification      
    
#------------------parameters-------------------------------------------------#
# fr_current:   the current feed rate
# lp_current:   the current laser power    
# median_temp:  the median temperature array of the previous layer
# layer:        the current cladding layer, must be integer
# is_fr:        determine whether the control target is feed rate or laser power    
# obj_temp:     objective temperature    
# activation_threshold: the activation temperature of fuzzy controller
# stable_threshold: if the error is less than stable_threshold, temperoraly stop updating the domain partition 
#      
#-----------------------------------------------------------------------------#
#    fuzzy_control = False

# the probability list of the error and derrordl, error_p and derrordl_p
# the probability list should have the same dimension as the domain partition     
    error_p    = [0., 0., 0., 0., 0.]
    derrordl_p = [0., 0., 0., 0., 0.]    

# initilize the laser power increment and feed rate increment 
    lp_incr = 0.
    fr_incr = 0.    
    
# determine whether the fuzzy controller is activated or no    
    if(median_temp[layer]<activation_threshold):
        dp_error.append(dp_error[-1].copy())
        dp_derrordl.append(dp_derrordl[-1].copy())
        dp_lp.append(dp_lp[-1].copy())
        dp_fr.append(dp_fr[-1].copy())
        return error_p, derrordl_p, lp_incr, fr_incr, dp_error, dp_derrordl
    
# if fuzzy control is activated, then calculate the error and derrordl
    error    = median_temp[layer] - obj_temp
    derrordl = median_temp[layer] - median_temp[layer-1]
    
# domain partion of the inputs(error and derrordl) and output(feedrate or laser power)
#    dp_error    = [[-50.,  -25., 0., 25.,  50.]]
#    dp_derrordl = [[-50.,  -25., 0., 25.,  50.]]
#    dp_fr       = [[-5.,   2.5,  0., 2.5,  5.]]
#    dp_lp       = [[-0.1, -0.05, 0., 0.05, 0.1]]
    
# step1: fuzzification

    
# calculate the probability of error
    if(dp_error[layer][0]>=error):
        error_p[0] = 1.0
    elif(dp_error[layer][4]<=error):
        error_p[4] = 1.0
    else:
        i=0
        while(error>dp_error[layer][i]):
            if(error<=dp_error[layer][i+1]):
                error_p[i]   = (dp_error[layer][i+1]-error)/(dp_error[layer][i+1]-dp_error[layer][i])
                error_p[i+1] = 1-error_p[i]
                break
            else:
                i += 1
            
# calculate the probability of derrordl
    if(dp_derrordl[layer][0]>=derrordl):
        derrordl_p[0] = 1.0
    elif(dp_derrordl[layer][4]<=derrordl):
        derrordl_p[4] = 1.0
    else:
        i=0
        while(derrordl>dp_derrordl[layer][i]):
            if(derrordl<=dp_derrordl[layer][i+1]):
                derrordl_p[i]   = (dp_derrordl[layer][i+1]-derrordl)/(dp_derrordl[layer][i+1]-dp_derrordl[layer][i])
                derrordl_p[i+1] = 1-derrordl_p[i]
                break
            else:
                i += 1
  
# step2: inference rules
# the determine the control rules
# initialize the control rule                
    control_rules = [[0.]*5 for _ in range(5)]           
# assign values to control rule 
    for i in range(5):
        for j in range(5):
            control_rules[i][j] = min(error_p[i],derrordl_p[j])

#    print('control_rules = ', control_rules) # for test only
# step3: defuzzification 
# using center of gravity in defuzzificaiton 
# area/ weighted area initialization          
    area_sum      = 0.
    weighted_area = 0.
#    applied_rule  = 0 
# determine trapezoid base
    if is_fr_control:
        output_dp = dp_fr
        base = abs(dp_fr[layer][4]-dp_fr[layer][2]) 
    else:
        output_dp = dp_lp
        base = abs(dp_lp[layer][4]-dp_lp[layer][2])    


    print('')
#    print('layer = ', layer)
#    print('base = ', base)
    print('dp_error = ', dp_error[-1])
    print('dp_derrordl = ', dp_derrordl[-1])
    print('error_p = ', error_p)
    print('derrordl_p = ', derrordl_p)                   
# calculate area and weighted area
    for i in range(5):
        for j in range(5):
            if((i+j)<=2):
                applied_rule = 0
            elif((i+j)==3):
                applied_rule = 1
            elif((i+j)==4):
                applied_rule = 2
            elif((i+j)==5):
                applied_rule = 3
            else:
                applied_rule = 4
                
            area          = (2-control_rules[i][j])*control_rules[i][j]*base/2
            area_sum      += area
            weighted_area += area*output_dp[layer][applied_rule]
            
# for test only print the variables            
#            print('')
#            print('i,j = ',i,j)
#            print('area = ', area)
#            print('control_rules[i][j] = ', control_rules[i][j])
#            print('area_sum = ', area_sum)
#            print('weighted_area = ', weighted_area)
#            print('applied_rule  = ', applied_rule)
    
# step4: update the domain partition
# If the error is less than the trheshold and the production of error and derrordl is less than 0,
# stop updating the domain partition. 
# Otherwise keep updating the domain partition.           

    if(abs(error)<stable_threshold[0])and(abs(derrordl)<stable_threshold[1]):
        dp_error.append(dp_error[-1].copy())
        dp_derrordl.append(dp_derrordl[-1].copy())
        dp_lp.append(dp_lp[-1].copy())
        dp_fr.append(dp_fr[-1].copy())
        return error_p, derrordl_p, lp_incr, fr_incr, dp_error, dp_derrordl
    else:
# keep updating the domain partion until the domain partition lower bound
# update the error domain partition     
        dp_error_update = dp_error[-1].copy()
        if(error>dp_error_update[0]) and (error<dp_error_lb):
# update the dp_error based on derrordl_p            
            dp_error_update[0] = error*0.4+dp_error_update[0]*0.6
#            dp_error_update[0] = derrordl_p[4]*dp_error_update[0]+derrordl_p[3]*error+derrordl_p[2]*dp_error_lb
#            dp_error_update[0] = derrordl_p[4]*dp_error_update[0]+derrordl_p[3]*error+derrordl_p[2]*dp_error_lb 
#            dp_error_update[0] = error
            dp_error_update[1] = dp_error_update[0]/2
        elif(error<0) and (error>dp_error_lb):
            dp_error_update[0] = dp_error_lb
            dp_error_update[1] = dp_error_lb/2
        if(error<dp_error_update[4]) and (error>dp_error_ub):
            dp_error_update[4] = error
            dp_error_update[3] = error/2
        elif(error>0) and (error<dp_error_ub):
            dp_error_update[4] = dp_error_ub
            dp_error_update[3] = dp_error_ub/2
        dp_error.append(dp_error_update)   
# update the derrordl domain partion
        dp_derrordl_update = dp_derrordl[-1].copy()
        if(derrordl>dp_derrordl_update[0]) and (derrordl<dp_dedl_lb):
            dp_derrordl_update[0] = derrordl
            dp_derrordl_update[1] = derrordl/2
        elif(derrordl<0) and (derrordl>dp_dedl_lb):
            dp_derrordl_update[0] = dp_dedl_lb
            dp_derrordl_update[1] = dp_dedl_lb/2
        if(derrordl<dp_error_update[4]) and (derrordl>dp_dedl_ub):
            dp_derrordl_update[4] = derrordl
            dp_derrordl_update[3] = derrordl/2
        elif(derrordl>0) and (derrordl<dp_dedl_ub):
            dp_derrordl_update[4] = dp_dedl_ub
            dp_derrordl_update[3] = dp_dedl_ub/2            
        dp_derrordl.append(dp_derrordl_update)
                        
#        dp_error.append(dp_error[-1])
#        dp_derrordl.append(dp_derrordl[-1])
        dp_lp.append(dp_lp[-1].copy())
        dp_fr.append(dp_fr[-1].copy())
        
#    dp_error.append(dp_error[-1])
#    dp_derrordl.append(dp_derrordl[-1])
#    dp_lp.append(dp_lp[-1])
#    dp_fr.append(dp_fr[-1])

    
    if(is_fr_control):
        fr_incr = weighted_area/area_sum
        lp_incr = 0.
        
    else:
        lp_incr = weighted_area/area_sum
        fr_incr = 0.
        
        
    
#    print('area_sum = ', area_sum)
#    print('weighted_area = ', weighted_area)
#    print('lp_incr = ',weighted_area/area_sum)
#    lp_incr = 0
    
    return error_p, derrordl_p, lp_incr, fr_incr, dp_error, dp_derrordl

"-----------------------------------------------------------------------------"
#
#
#
"_____________________________________________________________________________"

# just for test p
#@profile
def energy_balance(geometry_size, nodal_info, t_i, fr, lp, sample_interval,\
                   clad_geometry='thinwall',is_fuzzy_control=True, is_fr_control=True, R=1.541):
# fr: initial feed rate; the passed-in feed rate may change if feed rate control is adopted.
# lp: initial laser power    
# the defualt set is using fuzzy controller and the control variable is feed rate.
# the control objective is claimed inside the enegry_balance funciton.       
    
    SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width,SHM1,SHM2,CHM,LM,WM = geometry_size
    node_dynamic_info, node_connected, node_distance, node_coordinate, node_length, node_volume, node_cs, nodes_in_layer = nodal_info
 
# just for test purpose
# print the cps usage    
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    
# function start time
    f_start = time.time()

# substrate nodes    
    x_nodes_S = int(SubL/LM+1)
    y_nodes_S = int(SubH1/SHM1+SubH2/SHM2+1)
    z_nodes_S = int(SubW/WM+1)
# clad nodes
    x_nodes_C = int(CladL/LM+1)     
    y_nodes_C = int(CladH/CHM+1)
    z_nodes_C = int(CladW/WM+1)    
# substrate nodes
    nodes_S = int(x_nodes_S*y_nodes_S*z_nodes_S)
# clad nodes
#    nodes_C = int((x_nodes_C*z_nodes_C - (x_nodes_C-2*Width/LM-2)*(z_nodes_C-2*Width/WM-2))*y_nodes_C)
# interface nodes
    interface_nodes = nodes_in_layer[0]
    nodes_I = len(interface_nodes)   
# total number of nodes
    total_nodes = len(node_length)
# clad nodes
#    nodes_C = total_nodes-nodes_S+nodes_I     

# nodes in substrate
    substrate_nodes = np.arange(1,nodes_S+1)
    substrate_nodes = np.setdiff1d(substrate_nodes,interface_nodes)
    substrate_nodes = list(substrate_nodes)
    
# nodes in clad
#    clad_nodes = np.arange(nodes_S+1,total_nodes)
# cladded nodes
    cladded_nodes    = []
#    clad_top_surface = nodes_in_layer[0]
    clad_top_surface = []
    cladding_nodes   = nodes_in_layer[1]     

# direction array for clad nodes; 
# because the film condition in direction of y positive depends on whether the node above is active or not     
#    direction1 = np.array([0,1,3,4,5])
#    direction2 = np.array([0,1,4,5])
# directions lsit for substrate and clad nodes
    directions_substrate        = range(6)
    directions_interface        = [0,1,3,4,5]
    directions_cladded_nodes    = range(6)
    directions_clad_top_surface = [0,1,3,4,5]
    directions_cladding_nodes   = [0,1,3,4,5] 
#    horizontal_directions       = [0,1,4,5]
    
# total layers
# assuming CHM equals to layer height
    layer_height = CHM    
    total_layers = int(CladH/layer_height)

# Initialize output arrays and parameters
# laser track center coordinates on horizontal plane
# initial laser track center coordinates, x0, z0    
    if clad_geometry == 'thinwall':
        xc0 = 0.5*(SubL-CladL)
        zc0 = 0.5*SubW
        xc1 = 0.5*(SubL+CladL)
        zc1 = 0.5*SubW
    elif clad_geometry == 'square block':
        xc0 = 0.5*(SubL-CladL+Width)
        zc0 = 0.5*(SubW-CladW+Width)
        xc1 = 0.5*(SubL+CladL-Width)
        zc1 = 0.5*(SubW+CladW-Width)



# if the clad is square, then initial coordinates are:
#    xc0 = 0.5*(SubL-CladL+Width)
#    zc0 = 0.5*(SubW-CladW+Width)
# the laser track center coordinates on the opposite corner    
#    xc1 = 0.5*(SubL+CladL-Width)
#    zc1 = 0.5*(SubW+CladW-Width)

# if the clad is thinwall, the initial coordinates are:
#    xc0 = 0.5*(SubL-CladL)
#    zc0 = 0.5*SubW 
# the coordinate on the other side of thinwall
#    xc1 = 0.5*(SubL+CladL)
#    zc1 = 0.5*SubW    

    
# laser spot center coordinates
    LSC = np.array([[xc0,zc0]])   

# energy lose by convection and radiation
    energy_lose = np.array([])

# laser energy input; the data in this array is collected per # increment
    laser_energy = np.array([])

# laser power on each layer: the data in this array is collected at the end of each layer
    lp_layer = np.array([])   
# feed rate on each layer; the data in this array is collected at the end of each layer   
    fr_layer = np.array([])   
# median temperature of each layer
    median_temperature = np.array([])    
    
# conduction combined; just for test purpose
#    conduction_combined = np.array([])    
#    conduction_s = np.array([])
#    conduction_c = np.array([])
#    conduction_i = np.array([])
    
# node in laser spot for test purpose
#    node_in_laser_spot = np.array([])    

# Initialize the clad top surface y coordinates
# initial clad top surface to SubH and 0.
# 0. means the laser is not applied to the particular location yet;
# 1. means the laser is already apllied to the location.    
    SubH = SubH1 + SubH2
    CTS = np.array(x_nodes_C*[z_nodes_C*[[SubH,0.]]])
    
# initialize the layer number to 1    
    layer = 1 
# initialize the laser power with the passed in value
    lp_current = lp
#initialize the feed rate with the passed in value
    fr_current = fr    
# laser spot direction
    ls_dir = 1
# current dwell time
    dwell_tc = 0.
# dwell time    
    dwell_t  = 2.     

#-----------------------------------------------------------------------------#
# fuzzy control parameters    
    obj_temp = 600.
    activation_threshold = 400.
    stable_threshold = [15.,10.]

# fuzzy control domain partition    
    dp_error    = [[-400,-200,0,25,50]]
    dp_derrordl = [[-30,-15,0,15,30]]
    
    dp_fr = [[-1,-.5,0,.5,1]]
    dp_lp = [[0.06,0.03,0,-0.04,-0.08]]    
#-----------------------------------------------------------------------------#    

# nodal_T: nodal temperature
# n_T: nodal temperature from last increment    
    nodal_T = np.array([total_nodes*[25.]])
    nodal_T[0] = node_dynamic_info[:,0]
# nodal temperature from last increment
    T_li = np.copy(nodal_T[0])

# nodal_A: node active
    nodal_A = np.array([total_nodes*[0.]])
    nodal_A[0] = node_dynamic_info[:,1]
# nodal activation from last increment    
    A_li = np.copy(nodal_A[0])    
    
# nodal internal energy density
# by assuming initial temperature is 25.C, initial internal energy density is 1.18e10   
#    nodal_E = np.array([total_nodes*[1.18e10]])
# nodal internal enegry density from last increment
#    E_li = np.copy(nodal_E[0]) 
    n_E = np.array(total_nodes*[1.18e10])

# Assuming the enviromental temperature is 25.C
    T_air = 25.
    
# internal energy array
    internal_energy = energy_T()

# density of 316L
    rou = 7.95e-9       

# heat flux parameters
    p = np.array([334480., -9069.4, -61689., 16712., 29465., -12476., -17916.])    

# claim conductivity temperature array   
    k_T = [20.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.,\
           900., 1000., 1100., 1200., 1300., 1385., 1450., 1500., 1600.]

# claim the values of conductivity respective to the temperatures
    k_v = [13.4, 15.5, 17.6, 19.4, 21.8, 23.4, 24.5, 25.1, 27.2, \
           27.9, 29.1, 29.3, 30.9, 31.1, 31.0, 28.5, 29.5, 30.5]   

# range of the k_v and k_T
    index = range(17)

# range of the internal energy array in T_next function
    range_Tnext = range(15)   
 
# claim heat capacity temperature array   
    cp_T = [20.,   100.,  200.,  300.,  400.,  500.,  600.,  700.,\
            800.,  900.,  1000., 1100., 1200., 1300., 1385., 1450.]

# claim the values of heat capacity respective to the temperatures
    cp_v = [470e6, 490e6, 520e6, 540e6, 560e6, 570e6, 590e6, 600e6, \
            630e6, 640e6, 660e6, 670e6, 700e6, 710e6, 720e6, 830e6]    

# calculate the nodal temperature for next increment    
# incr: time increment
# assuming the constant feed rate and constant laser power
# laser travel distance per layer = 2*(CladL+CladW-2*Width)
# if the clad is square
#    total_increment = int(2*(CladL+CladW-2*Width)/fr*total_layers/t_i)+1


# increment per layer
# if the clad is a square, the increment per layer is:
#    incre_per_layer = int(2*(CladL+CladW-2*Width)/fr/t_i)  
# if the clad is the thinwall, the increment per layer is:    
#    incre_per_layer = int((CladL/fr_current+dwell_t)/t_i)

# if the clad is thinwall:
#    total_increment = incre_per_layer*total_layers+1

# memory test 
#    memory_sample_rate = 500
#   
    
    
# film test    
#    counter1 = []
#    counter2 = []
#    substrate_el = []
#    cladside_el  = []
#    cladtop_el   = []
    incr = 0
    while(layer<=total_layers):
# t_c: current time
#        t_c = incr*t_i
# claim a nodal temperature array for next increment; initialize the array with last increment        
        n_T = np.copy(T_li)
# claim a nodal active array for next increment;
        n_A = np.copy(A_li)
# declare a nodal internal energy density array for next increment
#        n_E = np.copy(E_li)                
# laser spot center coordinate
        xc, zc = np.copy(LSC[-1])         
# max clad height at current increment         
        clad_height = SubH + layer_height*layer 
# clad height threshold, upper bound and lower bound        
#        clad_height_ub = clad_height + 0.001
#        clad_height_lb = clad_height - layer_height -0.001        
# reset the laser top surface array; 
        CTS[:,:,1] = 0.        
 
# energy lose in this increment
        film_incr = 0.
               
# laser energy received in this increment
        q_incr = 0.      
        
# for test
# clad surface layer film node counter
#        csl_counter = 0
# cladding layer film node counter
#        cl_counter  = 0
#        sub_el = 0.
#        cs_el  = 0.
#        ct_el  = 0.
# node in laser spot
#        n_LS = 0   
# conduction in this increment
#        cond_incr = 0.        
#        cond_c = 0.
#        cond_s = 0.
#        cond_i = 0.
 
# determine whether this is the first layer
        is_first_layer = False
        if layer==1:
            is_first_layer = True    
        
# calculate the energy balance for substrate nodes
# substrate nodes, other than interface nodes, have no laser applied on.
# heat transfer is composed of conduction from connected nodes and convection/radiation from air        
        for n in substrate_nodes:
# calculate the nodal temperature for next increment by looping through the connected node/air             
# n_film: heat transfer by film condition(convection/radiation)
            n_film = 0.
# n_cond: heat transfer by conduction
            n_cond = 0.
# T: nodal temperature of the current node
            T = T_li[n]          
# node_length = [dx,dy,dz]
# V: volum of the current node
#            V = node_volume[n]              
# c_n: connected node                       
            for c_n in directions_substrate:
# cs_area: cross section area = V/(node_length in particular direction)                 
#                cs_area = V/node_length[n][int(c_n/2)]
                if node_connected[n][c_n]:
#                    print('n, d = ', n, node_distance[n][c_n])
# T_conn: Temperature of the connected node
#                    T_conn = nodal_T[-1][node_connected[n][c_n]]
#                    n_cond += k(0.5*(T+T_conn))*(T_conn-T)/node_distance[n][c_n]*node_cs[n][c_n]
                    n_cond += cmp.k(0.5*(T+T_li[node_connected[n][c_n]]))*(T_li[node_connected[n][c_n]]-T)/node_distance[n][c_n]*node_cs[n][c_n]
#                    try:
#                        n_cond += k(0.5*(T+T_conn))*(T_conn-T)/node_distance[n][c_n]*node_cs[n][c_n]
#                    except:
#                        n_t = node_connected[n][c_n]
#                        print('n, c_n, d = ', n, c_n, node_distance[n][c_n])
#                        print('T_conn = ', T_conn)
#                        print('node_connected[n][c_n] = ', node_connected[n][c_n])
#                        print('nodal_A[-3][n_t] = ', nodal_A[n][n_t])
#                        print('n_d = ', node_distance[n_t])
#                        print('T_conn_pre1 = ', nodal_T[-2][n_t])
#                        print('T_conn_pre2 = ', nodal_T[-3][n_t])
#                        for nn in range(6):
#                            print(node_connected[n_t][nn], nodal_T[-3][node_connected[n_t][nn]])
                else:
                    n_film += cmp.film(T)*(T-T_air)*node_cs[n][c_n]
            
# calculate the nodal temperature for next increment                    
            film_incr += n_film
            n_E[n] += t_i*(n_cond-n_film)/node_volume[n]/rou     
            n_T[n] = T_next(n_E[n], internal_energy, cp_T, cp_v, range_Tnext)
#            sub_el += n_film
#            cond_incr += n_cond
#            cond_s += n_cond

# delete the temperory variables
            del c_n, n_film, n_cond, T


# calculate the heat transfer in interface nodes; no laser applied            
        for n in interface_nodes:
# n_film: heat transfer by film condition(convection/radiation)
            n_film = 0.
# n_cond: heat transfer by conduction
            n_cond = 0.
# T: nodal temperature of the current node
            T = T_li[n]          

# c_n: connected node                       
            for c_n in directions_interface:
                n_cond += cmp.k(0.5*(T+T_li[node_connected[n][c_n]]))*(T_li[node_connected[n][c_n]]-T)/node_distance[n][c_n]*node_cs[n][c_n]
            


# some interface have different nodal cross section area on oppsite directions; calculate the film condition
            for d in range(3):
                if node_cs[n][2*d] != node_cs[n][2*d+1]:
                    n_film += cmp.film(T)*(T-T_air)*(node_cs[n][2*d+1]-node_cs[n][2*d])
            
            if is_first_layer:
# laser energy input
                q = 0.                
# nodal coordinate
                x, y, z = node_coordinate[n]
# row and col of the current node             
                row = int((x-0.5*(SubL-CladL))/LM)
                col = int((z-0.5*(SubW-CladW))/WM) 
                r_n = math.sqrt((x-xc)**2.+(z-zc)**2.)
                if r_n <= R and abs(y-CTS[row][col][0])<0.001 and not CTS[row][col][1]:
# laser is only applied at one location with same row and col                     
                    CTS[row][col][1] = 1.
                    q = heat_flux(r_n,lp_current,p)*node_cs[n][2]
                    q_incr += q
                    CTS[row][col][0] += CHM
                if A_li[node_connected[n][2]]:                    
                    n_cond += cmp.k(0.5*(T+T_li[node_connected[n][2]]))*(T_li[node_connected[n][2]]-T)/node_distance[n][2]*node_cs[n][2]
                else:        
                    n_film += cmp.film(T)*(T-T_air)*node_cs[n][2]
                n_E[n] += t_i*(q+n_cond-n_film)/node_volume[n]/rou                 
            else:
                n_cond += cmp.k(0.5*(T+T_li[node_connected[n][2]]))*(T_li[node_connected[n][2]]-T)/node_distance[n][2]*node_cs[n][2]
                n_E[n] += t_i*(n_cond-n_film)/node_volume[n]/rou 
            
# calculate the nodal temperature for next increment                                            
            n_T[n] = T_next(n_E[n], internal_energy, cp_T, cp_v, range_Tnext)

# energy lose by film condition and conduction            
            film_incr += n_film
#            cond_incr += n_cond
# delete variables            
            del c_n, n_film, n_cond, T, d
            
# calculate the heat transfer in cladded nodes; no laser applied            
        for n in cladded_nodes:
# calculate the nodal temperature for next increment by looping through the connected node/air             
# n_film: heat transfer by film condition(convection/radiation)
            n_film = 0.
# n_cond: heat transfer by conduction
            n_cond = 0.
# T: nodal temperature of the current node
            T = T_li[n]          
# node_length = [dx,dy,dz]
# V: volum of the current node
#            V = node_volume[n]              
# c_n: connected node                       
            for c_n in directions_cladded_nodes:
# cs_area: cross section area = V/(node_length in particular direction)                 
#                cs_area = V/node_length[n][int(c_n/2)]
                if node_connected[n][c_n]:
                    n_cond += cmp.k(0.5*(T+T_li[node_connected[n][c_n]]))*(T_li[node_connected[n][c_n]]-T)/node_distance[n][c_n]*node_cs[n][c_n]
#                elif not A_li[node_connected[n][c_n]]:
#                    n_film += film(T)*(T-T_air)*node_cs[n][c_n]
                else:
                    n_film += cmp.film(T)*(T-T_air)*node_cs[n][c_n]

# the interface nodes have different nodal cross section area on upper surface and lower surface
#            if node_cs[n][2]!=node_cs[n][3]:
#                n_film += film(T)*(T-T_air)*(node_cs[n][3]-node_cs[n][2])
            
# calculate the nodal temperature for next increment                                
            film_incr += n_film
            n_E[n] += t_i*(n_cond-n_film)/node_volume[n]/rou     
            n_T[n] = T_next(n_E[n], internal_energy, cp_T, cp_v, range_Tnext)
#            cs_el += n_film
#            cond_incr += n_cond

# delete the temperory variables
            del c_n, n_film, n_cond, T


# calculate the energy balance equation for interface nodes
        for n in clad_top_surface:
# T: nodal temperature of the current node
            T = T_li[n]
# set defualt value of is_laser_on to false          
#            is_laser_on = False                                    
# set the heat flux energy to 0.            
            q = 0.
# n_film: heat transfer by film condition(convection/radiation)
            n_film = 0.
# n_cond: heat transfer by conduction
            n_cond = 0.
# nodal coordinate
            x, y, z = node_coordinate[n]
# n_above: node above the current node                
            n_above = node_connected[n][2] 
# n_below: node below the current node            
            n_below = node_connected[n][3]
            
# nodal temperature above
            T_above = T_li[n_above]            

# cs_area_h: horizontal cross section area; the cross section area connected to node below
#            cs_area_horizontal = node_length[n][0]*node_length[n][2]
# laser is only applied on interface nodes on first layer
#            if is_first_layer:                                        
# row and col of the current node             
            row = int((x-0.5*(SubL-CladL))/LM)
            col = int((z-0.5*(SubW-CladW))/WM) 
            r_n = math.sqrt((x-xc)**2.+(z-zc)**2.)
            if r_n <= R and abs(y-CTS[row][col][0])<0.001 and not CTS[row][col][1]:
# laser is only applied at one location with same row and col                     
                CTS[row][col][1] = 1.
                q = heat_flux(r_n,lp_current,p)*node_cs[n][2]
                q_incr += q
#                if CTS[row][col][0]<clad_height:
                CTS[row][col][0] += CHM
                        
# calculate film condiction                
#            if not A_li[n_above]:
#                  n_film += film(T)*(T-T_air)*cs_area_horizontal
#                n_film += film(T)*(T-T_air)*node_cs[n][2]
#                V = node_volume[n]
            if A_li[n_above]:
#                node_volume[n] = node_distance[n][3]/2.*node_cs[n][3] + \
#                                 CHM/2.*node_cs[n][2]
                n_cond += cmp.k(0.5*(T+T_above))*(T_above-T)/node_distance[n][2]*node_cs[n_above][2]
            else:
                n_film += cmp.film(T)*(T-T_air)*node_cs[n][2]

                      
            for c_n in directions_clad_top_surface:
# cs_area: the cs_area is the vertical area connected to the node in x and z direction
# the cs_area is not the actual area, but it is easier to code in this way and has the same result as the actual cross section area
# the actual cross section area should be:
# cs_area_x: SHM1/2*WM+CHM/2*node_length[n_above][2] or SHM1/2*WM
# cs_area_z: SHM1/2*LM+CHM/2*node_length[n_above][0] or SHM1/2*LM               
# T_conn: Temperature of the connected node
                if node_connected[n][c_n]:
                    T_conn = T_li[node_connected[n][c_n]]
                    n_cond += cmp.k(0.5*(T+T_conn))*(T_conn-T)/node_distance[n][c_n]*node_cs[n][c_n]
                    
#                    L_conn = node_length[node_connected[n][c_n]][1]
#                    if L_conn==node_length[n][1]:    
#                        n_cond += k(0.5*(T+T_conn),k_T,k_v,index)*(T_conn-T)/node_distance[n][c_n]*node_cs[n][c_n]
#                    else:
#                        n_cond += k(0.5*(T+T_conn),k_T,k_v,index)*(T_conn-T)/node_distance[n][c_n]*node_cs[n][c_n]/node_length[n][1]*L_conn
#                        n_film += film(T)*(T-T_air)*node_cs[n][c_n]/node_length[n][1]*CHM/2.
                else:
                    n_film += cmp.film(T)*(T-T_air)*node_cs[n][c_n]
#            cond_i += n_cond
# nodal temperature of interface node for next increment                
            n_E[n] += t_i*(q+n_cond-n_film)/node_volume[n]/rou     
            n_T[n] = T_next(n_E[n], internal_energy, cp_T, cp_v, range_Tnext)
# energy lose by film condition            
            film_incr += n_film
#            cond_incr += n_cond
# delete the temperory variables
            del row, col, r_n, c_n, T, q, n_film, n_cond, x,y,z, n_above, T_above, n_below


# calculate the energy balance equation for Clad nodes
        for n in cladding_nodes:

# nodal coordinate
            x, y, z = node_coordinate[n]

# the node above
#            n_above = node_connected[n][2]
# n_film: heat transfer by film condition(convection/radiation)
            n_film = 0.
# n_cond: heat transfer by conduction
            n_cond = 0.  
            
# T: nodal temperature of the current node
            T = T_li[n]
# V: volum of the current node
#            V = node_volume[n]            
# set the heat flux energy to 0.            
            q = 0.           


# determine whether the node has laser applied on           
# determine whether the laser is inside the laser spot           
# set defualt value of is_laser_on to false          
            is_laser_on = False                
# row and col of the current node             
            row = int((x-0.5*(SubL-CladL))/LM)
            col = int((z-0.5*(SubW-CladW))/WM)
            r_n = math.sqrt((x-xc)**2.+(z-zc)**2.)
            if r_n<= R and abs(y-CTS[row][col][0])<0.001 and not CTS[row][col][1]:
# laser is only applied at one location with same row and col                     
                CTS[row][col][1] = 1.
                q = heat_flux(r_n,lp_current,p)*node_cs[n][2]
                q_incr += q
                is_laser_on = True

# determine the heat transfer condition for inactive clad nodes
# skip the loop if the node is inactive and has no laser applied            
            if not A_li[n] and not is_laser_on:
                continue
            elif not A_li[n] and is_laser_on:
# laser energy recieved = laser energy density*horizontal area
# because the node is ianctive, the nodal area and volume is half of the total voume                 
#                q = heat_flux(r_n,lp_cureent,p)*node_cs[n][2]
                n_E[n] += t_i*q/node_volume[n]/rou     
                n_T[n] = T_next(n_E[n], internal_energy, cp_T, cp_v, range_Tnext)
#                q_incr += q
# activate the node if the nodal temperature is above 1385C                
#                if n_T[n]>1385.:
#                    n_A[n] = 1.
# to mimic the simulation in abaqus, activate the inactive node in laser spot even if the nodal temperature is less than 1385 
                n_A[n] = 1.      
# add one layer to the clad top surface at current location                  
                if CTS[row][col][0]<clad_height:
                    CTS[row][col][0] += CHM
                continue

# clad top surface cannot be larger than clad height             
            if CTS[row][col][0]>clad_height:
                CTS[row][col][0] = clad_height
                    
# determine the heat transfer condition for active clad nodes            
# calculate the film condition and conduction depending on whether the node above is active or not            
#                V = node_volume[n]                
#                n_cond += k(0.5*(T+T_li[n_above]),k_T,k_v,index)*(T_li[n_above]-T)/node_distance[n][2]*node_cs[n][2]

# convection and radiation in y positive direction
            n_film += cmp.film(T)*(T-T_air)*node_cs[n][2]
#            cl_counter += 1
# calculate the heat transfer in direction other than y positive
            for c_n in directions_cladding_nodes:
                if A_li[node_connected[n][c_n]]:
# T_conn: Temperature of the connected node
                    T_conn = T_li[node_connected[n][c_n]]                    
                    n_cond += cmp.k(0.5*(T+T_conn))*(T_conn-T)/node_distance[n][c_n]*node_cs[n][c_n]
                else:
                    n_film += cmp.film(T)*(T-T_air)*node_cs[n][c_n]

# calculate the nodal temperature for next increment                    
            n_E[n] += t_i*(q+n_cond-n_film)/node_volume[n]/rou     
            n_T[n] = T_next(n_E[n], internal_energy, cp_T, cp_v, range_Tnext)
#            cond_incr += n_cond
#            cond_c += n_cond

# energy lose
            film_incr += n_film
# laser energy received 
#            q_incr += q            

# delete temperory variables
            del row, col, r_n, c_n, T, q, n_film, n_cond, x,y,z, is_laser_on


# calculate the laser spot center and layer number for next increment
            
#*square*#            
# start from the initial position, check each case in clockwise direction 
        current_layer = layer                                                   # set this layer as current layer
#        xc_next, zc_next, layer = clockwise_square(xc,zc,layer,xc0,xc1,zc0,zc1,fr,t_i)

        if clad_geometry == 'thinwall':
            xc_next, zc_next, layer, ls_dir, dwell_tc = \
            zigzag(xc,zc,layer,ls_dir,dwell_tc,dwell_t,xc0,xc1,zc0,zc1,fr_current,t_i,cladding_direction='x')
        elif clad_geometry == 'square block':
            xc_next, zc_next, layer = clockwise_square(xc,zc,layer,xc0,xc1,zc0,zc1,fr,t_i)

#*thinwall*# 
        
        
# append the laser spot center array for next increment               
        LSC = np.append(LSC,[[xc_next,zc_next]],axis=0)
# copy the temperature and activation array for next increment
        T_li = np.copy(n_T)
        A_li = np.copy(n_A)

# if 'layer' is not the current layer, add more nodes to cladded nodes
        if current_layer != layer:
            print(' ')
            print('current_layer, layer = ', current_layer, layer)
            print('incr, sample_interval = ', incr, sample_interval)
            if layer>2:
                cladded_nodes.extend(nodes_in_layer[layer-2])
            for n in clad_top_surface:
                if not A_li[n]:
                    node_volume[n] = node_distance[n][3]/2.*node_cs[n][3] + \
                                     CHM/2.*node_cs[n][2]
            clad_top_surface = nodes_in_layer[layer-1]
#            for n in clad_top_surface:
#                for d in horizontal_directions:
#                    node_cs[n][d] = node_cs[n][d]/node_length[n][2]*CHM
#                    node_cs[n][d] = node_cs[n][d]*2
            if layer<=total_layers:
                cladding_nodes = nodes_in_layer[layer]
            
            temp = np.median(n_T[(nodes_S+nodes_I*(current_layer-1)):(nodes_S+nodes_I*current_layer)])
            median_temperature = np.append(median_temperature,temp)
            
            if is_fuzzy_control:
                error_p, derrordl_p, lp_incr, fr_incr, dp_error, dp_derrordl = \
                fuzzy_control(lp_current, fr_current, median_temperature, current_layer-1, is_fr_control,
                              obj_temp, activation_threshold, stable_threshold,
                              dp_error,dp_derrordl,dp_fr,dp_lp)
# update the laser power and feed rate           
                lp_current += lp_incr
                fr_current += fr_incr
                
            lp_layer = np.append(lp_layer,lp_current)
            fr_layer = np.append(fr_layer,fr_current)

            print('lp_layer = ' + '\n', lp_layer)
            print('fr_layer = ' + '\n', fr_layer)
#            print('lp_incr = ', lp_incr)
#            print('error_p    = ', error_p)
#            print('derrordl_p = ', derrordl_p)
            print('dp_error[-1]    = ', dp_error[-1])
            print('dp_derrordl[-1] = ', dp_derrordl[-1])
            print('median_temperature, layer, current_layer = ', temp, layer, current_layer)
            print('running time = ', time.time()-f_start)
            del temp
            print(psutil.cpu_percent())
            print(psutil.virtual_memory())
            
            
        
# append the nodal temperature array and nodal active array every sample_interval
        if incr%sample_interval==0:
            nodal_T = np.append(nodal_T,[n_T],axis=0)
            nodal_A = np.append(nodal_A,[n_A],axis=0)
            
            
            
#            nodal_E = np.append(nodal_E,[n_E],axis=0)
#            print(' ')
#            print('t = ',time.time()-f_start)
#            print(psutil.cpu_percent())
#            print(psutil.virtual_memory())
            
#        if incr%memory_sample_rate==0:
#            print(' ')
#            print('t = ',time.time()-f_start)
#            print(psutil.cpu_percent())
#            print(psutil.virtual_memory())
            
# append the energy lose and energy received array
        energy_lose  = np.append(energy_lose,film_incr)
        laser_energy = np.append(laser_energy,q_incr)

        incr += 1
#        cond_incr = cond_c+cond_s+cond_i
#        conduction_combined = np.append(conduction_combined,cond_incr)
#        conduction_c = np.append(conduction_c, cond_c)
#        conduction_s = np.append(conduction_s, cond_s)
#        conduction_i = np.append(conduction_i, cond_i)
#        node_in_laser_spot = np.append(node_in_laser_spot,n_LS)

# just for test purpose
# collect leaked memory 
#        collected_memory=gc.collect()
#        print(collected_memory)            
#   

#        if not incr%incre_per_layer:
#            i = int(incr/incre_per_layer)
#            temp = np.median(n_T[(nodes_S+nodes_I*(i-1)):(nodes_S+nodes_I*i)])
#            median_temperature = np.append(median_temperature,temp)
            
#            error_p, derrordl_p, lp_incr, fr_incr, dp_error, dp_derrordl = \
#            fuzzy_control(lp_current, fr_current, median_temperature, current_layer-1, is_fr_control,
#                          obj_temp, activation_threshold, stable_threshold,
#                          dp_error,dp_derrordl,dp_fr,dp_lp)
#            lp_current += lp_incr
#            fr_current += fr_incr
#            lp_layer = np.append(lp_layer,lp_current)
#            fr_layer = np.append(fr_layer,fr_current)
#            print(' ')
#            print('lp_layer = ' + '\n', lp_layer)
#            print('fr_layer = ' + '\n', fr_layer)
#            print('lp_incr = ', lp_incr)
#            print('error_p    = ', error_p)
#            print('derrordl_p = ', derrordl_p)
#            print('dp_error[-1]    = ', dp_error[-1])
#            print('dp_derrordl[-1] = ', dp_derrordl[-1])
#            print('median_temperature, layer, i = ', temp, layer, i)
#            print('running time = ', time.time()-f_start)
#            del temp, i
#            print(psutil.cpu_percent())
#            print(psutil.virtual_memory())
#        else:
#            print('increment = ', incr)
#            print('t = ', time.time()-f_start)

        # if incr>= 100:
        #     print('time with cmp = ', time.time()-f_start)
        #     break

#        mid_time = time.time()
#        if mid_time-f_start>=100.:
#            break
# just for test purpose
# print the cps usage   
#            print(psutil.cpu_percent())
#            print(psutil.virtual_memory())  # physical memory usage            
#            break
#        if layer>=30:
#            break
# delete the arrays and variable
        del n_T, n_A, xc, zc, xc_next, zc_next, clad_height, film_incr, q_incr


# function end time
    f_end = time.time()
# function run time
    f_run = f_end - f_start                    
    print('function runtime is', f_run)
#    print('length of node_s, nodes_i, nodes_c = ', len(substrate_nodes), len(interface_nodes), len(clad_nodes))                    
    return nodal_T, nodal_A, LSC , energy_lose, laser_energy, \
            median_temperature,lp_layer, fr_layer, dp_error, dp_derrordl

"-----------------------------------------------------------------------------"




def main():
#   geometry(SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width=3.,SHM1=1.,SHM2=5.,CHM=0.7,WM=1.,LM=1.)
#   nodal_info_initialization(geometry_size)
#   energy_balance(geometry_size, nodal_info, t_i, fr, lp, R=1.541)

# get the program running time
    start_time = time.time()
    
# geometry size       
    SubH1 = 1.
    SubH2 = 24.
    SubL  = 100.
    SubW  = 51.
    CladH = 41.5
    CladL = 80.
    CladW = 3.
    Width = 3.
    SHM1  = 1.
    SHM2  = 6.
    CHM   = 0.83
    LM    = 1.
    WM    = 1.
    
    

# laser power 1.7 kW; 
# feed rate 1000mm/min    
    t_i = 0.01
    fr  = 1000./60.
    lp  = 1.7
    sample_interval = 100       
    geo = geometry(SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width,SHM1,SHM2,CHM,LM,WM)
#    geometry(SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width=3.,SHM1=1.,SHM2=5.,CHM=0.7,LM=1.,WM=1.):
    thinwall_info = thinwall_initialization(geo)

# is fuzzy control
    fuzzy_control = True

# is feed rate control 
    fr_control = False    
    
    node_dynamic_info, node_connected, node_distance, node_coordinate, node_length, node_volume, node_cs, nodes_in_layer = thinwall_info
    
#    energy_balance(geometry_size, nodal_info, t_i, fr, lp, sample_interval,\
#                   clad_geometry='thinwall',is_fuzzy_control=True, is_fr_control=True, R=1.541)
    
    nodal_T, nodal_A, LSC, energy_lose, laser_energy, median_temperature, lp_layer, fr_layer, dp_error, dp_derrordl = \
    energy_balance(geo,thinwall_info, t_i, fr, lp, sample_interval, 'thinwall',fuzzy_control,True)
    # print('LSC = ', LSC)
    print('median_temperature = ', median_temperature)
    with open('thinwall_test_fr_control_lp1.7.npy','wb') as f:
        np.save(f,nodal_T)
        np.save(f,nodal_A)
        np.save(f,LSC)
        np.save(f,energy_lose)
        np.save(f,laser_energy)
        np.save(f,median_temperature)
        np.save(f,lp_layer)
        np.save(f,fr_layer)        
        np.save(f,dp_error)
        np.save(f,dp_derrordl)

    
#    nodal_T1, nodal_A1, LSC1, energy_lose1, laser_energy1,conduction_combined1,conduction_c1, conduction_s1,conduction_i1 = energy_balance(geo,block_info, t_i, fr, lp)
    
    end_time = time.time()
    total_time = end_time - start_time
    print('program ended at: ', end_time)
    print('program runtime: ', total_time)
    
    return 0
#    ax.contour3D(x, y, z_o, 50, cmap='binary')
#    plt.legend(["python simulation", "Abauqs simulation"])
#    plt.show()


if __name__ == "__main__":
    main()


#   xc,zc,xc0,xc1,zc0,zc1,fr,direction,dwell_t,dwell_tc,t_i,layer
""" 
    
    geometry(SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width=3.,SHM1=1.,SHM2=5.,CHM=0.7,LM=1.,WM=1.)
    geo = geometry(SubH1,SubH2,SubL,SubW,CladH,CladL,CladW,Width,SHM1,SHM2,CHM,LM,WM)
   
    xc_next, zc_next, layer, dwell_tc, direction = zigzag(xc,zc,layer,direction,dwell_tc,dwell_t,xc0,xc1,zc0,zc1,fr,t_i)
    xc = 10.
    zc = 22.5
    xc0 = 10.
    zc0 = 22.5
    xc1 = 40.
    zc1 = 22.5
    fr = 1000./60.
    direction = 1
    dwell_t = 2.
    dwell_tc = 0.
    t_i = 0.005
    layer = 1
    
    LSC = np.array([[xc0,zc0,0,1,0.]])
    
    for i in range(2000):
        p1,p2,p3,p4,p5 = LSC[-1]
        LSC_next = zigzag(p1,p2,p3,p4,p5,dwell_t,xc0,xc1,zc0,zc1,fr,t_i)
        LSC = np.append(LSC,[LSC_next],axis=0)
    
    
# extract median temperature of each layer
    median_temperature = np.array([])
    for i in range(1,5):
        temp = np.median(clad_T[680*i,324*(i-1):324*i])
        median_temperature = np.append(median_temperature,temp)
        
    t_start = time.time()
    R = 1.541
    x = 23.5
    z = 1.5
    x0 = 22.5
    z0 = 1.1
    k = 0
    
    t_start = time.time()
    for i in range(1000):
        temp = T_next(25, 1000., 0.5, internal_energy)
    print('t = ', time.time()-t_start)
    
    t_start = time.time()
    for i in range(1000):
        arr = np.array([[1,2,3,4,5]*1000])
    print('t = ', time.time()-t_start)
    
    
    for i in range(1000):
        if math.sqrt((x-x0)**2.+(z-z0)**2.)<R:
            k += 1
        else:
            k -= 1
    print('t = ', time.time()-t_start)
    print('k = ', k)


# kb test
    temps = [25,100,205,333,400,542,623,777,812,911,1023,1122,1200,1323,1467,1555,1600,1777]
    for t in temps:
        print('kb = ', kb(t,k_T,k_v))
        print('k  = ', k(t,k_T,k_v,k_range))
        print(' ')
    
    t_start = time.time()
    for i in range(100000):
        for t in temps:
            test = kb(t,k_T,k_v)
    print('t = ',time.time()-t_start)
    
# summation test
    test1 = np.array(100000*[[1,2,3,4,5,6]])
    test2 = np.array(100000*[0.])
    s_time = time.time()
    for i in range(10000):
        for j in range(100000):
            test2 = 0.
            for k in range(6):
                test2 += test1[j,k]
    print('t = ',time.time()-s_time)
    
# memory test
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage 
    for j in range(5):
        for i in range(100000):
            a = i+2
            b = np.array([[1,2,3,4,5]])
            if a>100000:
                print(psutil.cpu_percent())
                print(psutil.virtual_memory())  # physical memory usage
                tracemalloc.take_snapshot()
                display_top(snapshot)
                
# loop test
    
    test1 = range(6)
    test2 = [0,1,2,3,4,5]
    test3 = np.arange(6)
    t_start = time.time()
    for i in range(100000):
        for j in test1:
            continue
    print('t = ', time.time()-t_start)

    with open('squareblock_test_lp1.7.npy','rb') as f:
        nodal_T2 = np.load(f)
        nodal_A2 = np.load(f)
        LSC = np.load(f)
        energy_lose = np.load(f)
        laser_energy = np.load(f)
        median_temperature = np.load(f)  
        lp_layer = np.load(f)
        fr_layer = np.load(f)
        dp_error = np.load(f)
        dp_dedl = np.load(f)

#-----------------------------------------------------------------------------#
simulation: thinwall_test parameters sets and domain partition:
**********
lp control
**********    
1.
parameter set
    obj_temp = 1150.
    activation_threshold = 950.
    stable_threshold = 20.
    is_fr_control = False
    dp_error    = [[-400,-200,0,25,50]]
    dp_derrordl = [[-30,-15,0,15,30]]
    dp_fr = [[-5,-2.5,0,2.5,5]]
    dp_lp = [[0.06,0.03,0,-0.04,-0.08]]    
    
domain partition
    if(error>dp_error_update[0]) and (error<dp_error_lb):        
        dp_error_update[0] = error*0.4+dp_error_update[0]*0.6
        dp_error_update[1] = dp_error_update[0]/2
    elif(error<0) and (error>dp_error_lb):
        dp_error_update[0] = dp_error_lb
        dp_error_update[1] = dp_error_lb/2
    if(error<dp_error_update[4]) and (error>dp_error_ub):
        dp_error_update[4] = error
        dp_error_update[3] = error/2
    elif(error>0) and (error<dp_error_ub):
        dp_error_update[4] = dp_error_ub
        dp_error_update[3] = dp_error_ub/2
        
2. same as 1 except
    dp_error    = [[-400,-200,0,25,50]]
    dp_derrordl = [[-30,-15,0,15,30]]
    dp_fr = [[-5,-2.5,0,2.5,5]]
    dp_lp = [[0.06,0.03,0,-0.04,-0.08]]    

3. same as 2 except CladH = 28-->42         
    
#-----------------------------------------------------------------------------#    


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
line1 = ax1.plot(layer,lp_layer,'r--',label='Laser power')
line2 = ax2.plot(layer,median_temperature,'b-',label='Median temperature')
ax1.legend(line1+line2,['Laser power','Median temperature'],loc='lower center')
ax1.set_xlabel('Layer')
ax1.set_ylabel('Laser power (kW)')
ax2.set_ylabel('Median temperature (C)')
for x,y in zip(layer,laser_layer):
    label = "{:.2f}".format(y)
    ax1.annotate(label,(x,y),textcoords="offset points", xytext=(0,10),ha='center')
for x,y in zip(layer,median_temperature):
    label = "{:.2f}".format(y)
    ax2.annotate(label,(x,y),textcoords="offset points", xytext=(0,10),ha='center')    

"""    