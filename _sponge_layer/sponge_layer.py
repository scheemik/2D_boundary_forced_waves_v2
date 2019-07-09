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

def sponge_profile(z, z_bottom, z_top, slope, max_coeff, H_sl):
    # initialize array of values to be returned
    values = 0*z
    # Find height of domain
    H = z_top - z_bottom
    # Find middle of sponge layer
    sp_t = z_bottom + H*H_sl/2
    # Add upper stratification
    values += 1 + tanh_(z, max_coeff-1, slope, sp_t)
    return values

# %%
