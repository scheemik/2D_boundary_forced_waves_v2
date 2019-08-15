# Parameters for measuring energy flux


import numpy as np

# Aspect ratio of display domain
aspect_ratio = 1.0

# Domain size
L_x = 0.5 # m
L_z = 0.5 # m, not including the sponge layer
x_interval = (0.0, L_x)
z_b, z_t = -L_z, 0.0
# Display limits for plot
x_limits=[0.0, 0.5]

# Boundary forcing parameters
# Characteristic stratification
N_0 = 1.0 # [rad/s]
# Angle of beam w.r.t. the horizontal
theta = np.pi/4
# Horizontal wavelength
lam_x = L_x / 3.0
# Horizontal wavenumber
k_x    = 2*np.pi/lam_x
# Vertical wavenumber = 2*pi/lam_z, or from trig:
k_z    = k_x * np.tan(theta)
# Oscillation frequency = N_0 * cos(theta), from dispersion relation
omega  = N_0 * np.cos(theta) # [s^-1]
# Oscillation period = 2pi / omega
T = 2*np.pi / omega
# Forcing amplitude modifier
forcing_amp = 2.0e-4
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
z_sb     = -2.0 #1.5

# Background profile parameters
bp_slope = 120
stair_top =  0.0 - 0.2    # m
stair_bot = -L_z + 0.23    # m
bump = 1.3              # N (rad s^-1)
bg_height = N_0         # N (rad s^-1)
