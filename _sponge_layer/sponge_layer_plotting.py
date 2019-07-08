"""
A script to test plotting Foran's background profiles without
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
sys.path.insert(0, './_sponge_layer')
from sponge_layer import sponge_profile

# Parameters
nx = 512
nz = 128
aspect_ratio = 3.0
Lx, Lz = (aspect_ratio, 1.)
z_b, z_t = (-Lz/2, Lz/2)

# Parameters to set a sponge layer at the bottom
nz_sp = 40          # number of grid points in z direction in sponge domain
sp_slope = -50.     # slope of tanh function in slope
max_sp   =  50.     # max coefficient for nu at bottom of sponge
H_sl     =  0.5     # height of sponge layer = 2 * H_sl * Lz
z_sb     = z_b-2*H_sl*Lz      # bottom of sponge layer

###############################################################################

# Create bases and domain
x_basis  = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
z_main   = de.Chebyshev('zm', nz, interval=(z_b, z_t), dealias=3/2)
z_sponge = de.Chebyshev('zs', nz_sp, interval=(z_sb, z_b), dealias=3/2)
z_basis  = de.Compound('z', (z_sponge, z_main))
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

###############################################################################

# Store profile in an array
sponge_array = sponge_profile(z, z_sb, z_t, sp_slope, max_sp, H_sl)

# Plots the background profile
plot_bgpf = True
if (plot_bgpf):
    font = {'size' : 12}
    plt.rc('font', **font)
    plot_z = np.array(z[0])
    plot_p = np.array(sponge_array[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax1 = plt.subplots(1,1)

        ax1.set_title('Test Profile')
        ax1.set_xlabel(r'sponge coefficient')
        ax1.set_ylabel(r'depth ($z$)')
        ax1.set_ylim([z_sb,z_t])
        ax1.plot(plot_p, plot_z, 'k-')

        plt.grid(True)
        plt.show()
# %%
