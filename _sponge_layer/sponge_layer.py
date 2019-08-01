"""
A script for the sponge profile functions

Modified by Mikhail Schee, June 2019

"""

###############################################################################

import numpy as np

# Functions to define a sponge coefficient profile

def tanh_(z, height, slope, center):
    # initialize array of values to be returned
    values = 0*z
    # calculate step
    values = 0.5*height*(np.tanh(slope*(z-center))+1)
    return values

def sponge_profile(z, z_spbot, z_bottom, slope, max_coeff):
    # initialize array of values to be returned
    values = 0*z
    # Find height of sponge layer
    H = z_bottom - z_spbot
    # Find 2/3 down the sponge layer
    sp_c = z_bottom - 2.0*H/3.0
    # Add upper stratification
    values += 1 + tanh_(z, max_coeff-1, slope, sp_c)
    return values

# %%
