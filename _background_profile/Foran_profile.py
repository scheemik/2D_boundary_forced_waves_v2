"""
A script to test plotting the background profiles without
having to run the whole Dedalus script.

Modified by Mikhail Schee, June 2019

"""

###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import time

from dedalus import public as de
from dedalus.extras import flow_tools
"""
import sys
sys.path.insert(0, './_background_profile')
from background_profile import rho_profile
from background_profile import N2_profile
"""

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
    values = (height*slope)/(2.0*(np.cosh(slope*(z-center)))**2.0)
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
        for i in range(n):
            c_i = z_b + (height/2.0 + i*height)
            values += cosh2(z, height, slope, c_i)
    return values

# Parameters
nx = 512
nz = 128
aspect_ratio = 4.0
Lx, Lz = (aspect_ratio, 1.)
z_b, z_t = (-Lz/2, Lz/2)

###############################################################################

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(z_b, z_t), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

###############################################################################

# Parameters to determine a specific staircase profile
n_layers = 0
slope = 100.0*(n_layers+1)
N_1 = 0.95
N_2 = 1.24
z_bot = -0.05 #z_b
z_top =  0.05 #z_t

# Store profile in an array
bgpf_array = Foran_profile(z, n_layers, z_bot, z_top, slope, N_1, N_2)
#bgpf_array2 = N2_profile(z, n_layers, val_bot, val_top, slope, z_bot, z_top)

# Plots the background profile
plot_bgpf = True
if (plot_bgpf):
    plot_z = np.array(z[0])
    plot_p = np.array(bgpf_array[0])
    #plot_N = np.array(bgpf_array2[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax1 = plt.subplots(1,1)
        #ax2 = ax1.twiny()

        ax1.set_title('Test Profile')
        ax1.set_xlabel(r'density ($\bar\rho$)')
        ax1.set_ylabel(r'depth ($z$)')
        ax1.set_ylim([z_b,z_t])
        ax1.plot(plot_p, plot_z, 'k-')

        #ax2.set_xlabel(r'frequency ($N^2$)')
        #ax2.plot(plot_N, plot_z, '-')
        plt.grid(True)
        plt.show()
# %%
