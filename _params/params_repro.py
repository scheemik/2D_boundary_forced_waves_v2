# Parameters for running a reproduction of Foran's/Ghaemsaidi's system
#   Runs faster, but coarser resolution

import numpy as np

# Stop times for the simulation
sim_time_stop = 25 # time units (t)
wall_time_stop = 60 # min

# Determine whether adaptive time stepping is on or off
adapt_dt = False

# Aspect ratio of display domain
aspect_ratio = 1.0

# number of layers
n_layers = 2

# Number of points in each dimension
n_x = 256
n_z = 512

# Domain size
L_x = 1.5 # m
L_z = 0.5 # m, not including the sponge layer
x_interval = (-L_x/3.0, 2.0*L_x/3.0)
z_b, z_t = -L_z, 0.0

# Background profile parameters
profile_slope = 200.0
N_1 = 0.95                  # The stratification value above the staircase
N_2 = 1.24                  # The stratification value below the staircase
stair_bot_1 = -0.30         # Bottom of staircase (not domain) for 1 layer
stair_bot_2 = -0.38         # Bottom of staircase (not domain) for 2 layer
stair_top   = -0.22         # Top of staircase (not domian)

# Boundary forcing parameters
#   Bounds of the forcing window
forcing_left_edge = -1.0*L_x/12.0
forcing_rightedge =  1.0*L_x/12.0
# Angle of beam w.r.t. the horizontal
theta = np.pi/4
# Horizontal wavelength
lam_x = forcing_rightedge - forcing_left_edge
# Horizontal wavenumber
k_x    = 2*np.pi/lam_x
# Vertical wavenumber = 2*pi/lam_z, or from trig:
k_z    = k_x * np.tan(theta)
# Other parameters
forcing_slope = 10 # for tanh window
forcing_amp   = 3.0e-4
