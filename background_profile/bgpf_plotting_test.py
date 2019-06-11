"""
Dedalus script for simulating internal waves through a vertically stratified
fluid with a density profile similar to a double-diffusive staircase.
Originally for 2D Rayleigh-Benard convection.

Modified by Mikhail Schee, June 2019

Usage:
    current_code.py DN1 DN2 DN3 DN4

Arguments:
    DN1		 Dimensionless number: Rayleigh
    DN2		 Dimensionless number: Prandtl
    DN3      Dimensionless number: Reynolds
    DN4      Dimensionless number: Richardson

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations have been non-dimensionalized.

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge.py` script in this
folder can be used to merge distributed analysis sets from parallel runs,
and the `plot_2d_series.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ conda activate dedalus 		# activates dedalus
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 merge.py snapshots
    $ mpiexec -n 4 python3 plot_2d_series.py snapshots/*.h5
    $ python3 create_gif.py

The 'mpiexec -n 4' can be ommited to run in series

The simulation should take a few process-minutes to run.

"""

import numpy as np
import matplotlib.pyplot as plt
import time
from background_profile import staircase

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
n_layers = 3
layer_ratio = 1
val_bot = 1.0
val_top = -1.0

# Background Profile (bgpf) as an NCC
bgpf = domain.new_field()
bgpf.meta['x']['constant'] = True  # means the NCC is constant along x
# Store profile in an array so it can be used for initial conditions later
bgpf_array = staircase(z, n_layers, layer_ratio, z_b, z_t, val_bot, val_top)
#bgpf_array2 = staircase(z, n_layers-1, layer_ratio, z_b, z_t, val_bot, val_top)
bgpf['g'] = bgpf_array
#problem.parameters['bgpf'] = bgpf  # pass function in as a parameter
del bgpf

# Plots the background profile
plot_bgpf = True
if (plot_bgpf):
    print(bgpf_array[0])
    vert = np.array(z[0])
    hori = np.array(bgpf_array[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax = plt.subplots(1,1)
        ax.set_title('Test Profile')
        ax.set_xlabel(r'density ($\bar\rho$)')
        ax.set_ylabel(r'depth ($z$)')
        ax.plot(hori, vert, '-')
        plt.grid(True)
        plt.show()
# %%
