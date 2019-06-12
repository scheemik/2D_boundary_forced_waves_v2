"""
A script to test plotting the background profiles without
having to run the whole Dedalus script.

Modified by Mikhail Schee, June 2019

"""

import numpy as np
import matplotlib.pyplot as plt
import time
# Next import relies on being in the same directory as background_profile.py
from background_profile import staircase
from background_profile import N_staircase

from dedalus import public as de
from dedalus.extras import flow_tools

# Parameters
nx = 40*7
nz = 40*2
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
n_layers = 2
layer_ratio = 1
val_bot = 1.0
val_top = -1.0

# Store profile in an array
bgpf_array = staircase(z, n_layers, layer_ratio, z_b, z_t, val_bot, val_top)
bgpf_array2 = N_staircase(z, n_layers, layer_ratio, z_b, z_t, val_bot, val_top)

# Plots the background profile
plot_bgpf = True
if (plot_bgpf):
    #print(bgpf_array[0])
    print(bgpf_array2[0])
    vert = np.array(z[0])
    hori = np.array(bgpf_array[0])
    hori2 = np.array(bgpf_array2[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax1 = plt.subplots(1,1)
        ax2 = ax1.twiny()

        ax1.set_title('Test Profile')
        ax1.set_xlabel(r'density ($\bar\rho$)')
        ax1.set_ylabel(r'depth ($z$)')
        ax1.plot(hori, vert, 'k-')

        ax2.set_xlabel(r'frequency ($N^2$)')
        ax2.plot(hori2, vert, '-')
        plt.grid(True)
        plt.show()
