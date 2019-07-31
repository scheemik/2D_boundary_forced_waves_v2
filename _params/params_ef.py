# Parameters for running a reproduction of Foran's/Ghaemsaidi's system
#   Runs faster, but coarser resolution

import numpy as np

# Stop times for the simulation
sim_period_stop = 2 # oscillation periods, time in seconds calculated below
wall_time_stop = 35 # min

# Determine whether adaptive time stepping is on or off
adapt_dt = False

# Number of points in each dimension
n_x = 256
n_z = 512

# Domain size
L_x = 0.5 # m
L_z = 0.5 # m, not including the sponge layer
x_interval = (0.0, L_x)
z_b, z_t = -L_z, 0.0

# Boundary forcing parameters
# Characteristic stratification
N_0 = 1.0 # [rad/s]
# Angle of beam w.r.t. the horizontal
theta = np.pi/4
# Horizontal wavelength
lam_x = L_x
# Horizontal wavenumber
k_x    = 2*np.pi/lam_x
# Vertical wavenumber = 2*pi/lam_z, or from trig:
k_z    = k_x * np.tan(theta)
# Oscillation frequency = N_0 * cos(theta), from dispersion relation
omega = N_0 * np.cos(theta) # [s^-1]
# Other parameters
forcing_amp   = 1.0e-4

# Calculate stop time
sim_time_stop = sim_period_stop * 2*np.pi / omega # time units (t)
