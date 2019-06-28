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

import sys
sys.path.insert(0, './_background_profile')
from background_profile import rho_profile
from background_profile import N2_profile

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
n_layers = -1
slope = 10.0*n_layers
val_bot = +0.1
val_top = -0.1
z_bot = z_b
z_top = z_t

# Store profile in an array
bgpf_array = rho_profile(z, n_layers, val_bot, val_top, slope, z_bot, z_top)
bgpf_array2 = N2_profile(z, n_layers, val_bot, val_top, slope, z_bot, z_top)

# Plots the background profile
plot_bgpf = True
if (plot_bgpf):
    plot_z = np.array(z[0])
    plot_p = np.array(bgpf_array[0])
    plot_N = np.array(bgpf_array2[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax1 = plt.subplots(1,1)
        ax2 = ax1.twiny()

        ax1.set_title('Test Profile')
        ax1.set_xlabel(r'density ($\bar\rho$)')
        ax1.set_ylabel(r'depth ($z$)')
        ax1.set_ylim([z_b,z_t])
        ax1.plot(plot_p, plot_z, 'k-')

        ax2.set_xlabel(r'frequency ($N^2$)')
        ax2.plot(plot_N, plot_z, '-')
        plt.grid(True)
        plt.show()
# %%
