# Parameters for running a reproduction of Foran's/Ghaemsaidi's system
#   Single layer

import numpy as np

# Aspect ratio of display domain
aspect_ratio = 1.0

# Domain size
L_x = 1.5 # m
L_z = 0.5 # m, not including the sponge layer
x_interval = (-L_x/3.0, 2.0*L_x/3.0)
z_b, z_t = -L_z, 0.0
# Display limits for plot
x_limits=[0.0, 0.5]

# Background profile parameters
profile_slope = 200.0
N_1 = 0.95                  # The stratification value above the staircase
N_2 = 1.24                  # The stratification value below the staircase
stair_bot = -0.30         # Bottom of staircase (not domain) for 1 layer
stair_top   = -0.22         # Top of staircase (not domian)

# Boundary forcing parameters
# Characteristic stratification
N_0 = N_1 # 1.0 [rad/s]
# Characteristic wavenumber
k   = 45 # [m^-1]
# Oscillation frequency = N_0 * cos(theta), from dispersion relation
omega = 0.67 # [rad s^-1]
# Angle of beam w.r.t. the horizontal
theta = np.arccos(omega/N_0) # [rad]
# Horizontal wavenumber
k_x   = k*omega/N_0 # [m^-1] k*cos(theta)
# Vertical wavenumber
k_z   = k*np.sin(theta) # [m^-1] k*sin(theta)
# Horizontal wavelength
lam_x = 2*np.pi / k_x # [m]
# Bounds of the forcing window
forcing_left_edge = x_limits[0] - lam_x/2.0 #-1.0*L_x/24.0
forcing_rightedge = x_limits[0] + lam_x/2.0 # 1.0*L_x/24.0

# Oscillation period = 2pi / omega
T = 2*np.pi / omega
# Other parameters
forcing_slope = 20 # for tanh window
# Forcing amplitude modifier
forcing_amp = 2.5e-4
# Forcing amplitude ramp (number of oscillations)
nT = 3.0

# Sponge layer parameters
# Number of grid points in z direction in sponge domain
nz_sp    =  40
# Slope of tanh function in slope
sp_slope = -5.0
# Max coefficient for nu at bottom of sponge
max_sp   =  50.0
# Bottom of sponge layer
z_sb     = -1.5
