"""
A script to test plotting the background profiles without
having to run the whole Dedalus script.

Modified by Mikhail Schee, June 2019

"""

###############################################################################

import numpy as np

# Functions to define a Foran N^2 profile

def tanh_(z, height, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate step
    values = 0.5*height*(np.tanh(slope*(z-center))+1)
    return values

def cosh2(z, height, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate step
    values = height/(np.cosh(slope*(z-center))**2.0)
    #values = (height*slope)/(2.0*(np.cosh(slope*(z-center)))**2.0)
    return values

def tanh_bump(z, height, slope, center, width):
    # initialize array of values to be returned
    values = 0*z
    # calculate sides of bump
    c_l, c_r = (center-width/2.0), (center+width/2.0)
    # add left side
    values += tanh_(z, height, slope, c_l)
    # add right side
    values += tanh_(z, height, -slope, c_r)
    return values

def Foran_profile(z, n, z_b, z_t, slope, N_1, N_2):
    # initialize array of values to be returned
    values = 0*z
    # Add upper stratification
    values += tanh_(z, N_1, slope, z_t)
    # Add lower stratification
    values += tanh_(z, N_2, -slope, z_b)
    # Find height of staircase region
    H = z_t - z_b
    # If there are steps to be added...
    if (n > 0):
        # calculate height of steps
        height = H / float(n)
        # calculate height of pseudo delta bumps as midpoint between N1 and N2
        bump_h = max(N_1, N_2) - 0.5*abs(N_1-N_2)
        for i in range(n):
            c_i = z_b + (height/2.0 + i*height)
            values += tanh_bump(z, bump_h, slope, c_i, 0.05)
            #values += cosh2(z, bump_h, slope, c_i)
    return values

# %%
