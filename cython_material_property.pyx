# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:28:32 2022

@author: Geng Li
"""
import cython


cdef double k_T[18] 
cdef double k_v[18]
cdef double cp_T[16]
cdef double cp_v[16]
        
# claim conductivity temperature array       
k_T[:] = [20.,  100.,  200.,  300.,  400.,  500.,  600.,  700.,  800., 
          900., 1000., 1100., 1200., 1300., 1385., 1450., 1500., 1600.]

# claim the values of conductivity respective to the temperatures    
k_v[:] = [13.4, 15.5, 17.6, 19.4, 21.8, 23.4, 24.5, 25.1, 27.2, 
          27.9, 29.1, 29.3, 30.9, 31.1, 31.0, 28.5, 29.5, 30.5]  

# claim heat capacity temperature array   
cp_T[:] = [20.,   100.,  200.,  300.,  400.,  500.,  600.,  700.,
           800.,  900.,  1000., 1100., 1200., 1300., 1385., 1450.]

# claim the values of heat capacity respective to the temperatures
cp_v[:] = [470e6, 490e6, 520e6, 540e6, 560e6, 570e6, 590e6, 600e6,
           630e6, 640e6, 660e6, 670e6, 700e6, 710e6, 720e6, 830e6]    

# defile heat capacity 
def cp(double T):
# find the heat capacity value of the current temperture     
    if(T>=1450.):
        return 830e6
    elif(T<=20.):
        return 470e6

# calculate the heat capacity value by linear interpolation    
# specify the range of the indexes will make the calculation faster
# if the material is changed, make sure to change the index range too
    for i in range(0,15):
        if(T<cp_T[i+1]):
            return cp_v[i]+(cp_v[i+1]-cp_v[i])/(cp_T[i+1]-cp_T[i])*(T-cp_T[i])


# define conductivity
def k(double T):
# find the conductivity value of the current temperture     
    if(T>=1600.):
        return 30.5
    elif(T<=20.):
        return 13.4

# index is the range(17); 17 is the index of the last element in k_T and k_v array    
    for i in range(0,17):
        if(T<k_T[i+1]):
            return k_v[i]+(k_v[i+1]-k_v[i])/(k_T[i+1]-k_T[i])*(T-k_T[i])
        

# define flim coefficient
def film(double T):
#    if(T<=0):
#        return 0.0121
    if T<3000.:
# (0.1921-0.0121)/3000. = 6e-5        
        return 0.0121+T*0.0006
    else:
        return 0.1921






