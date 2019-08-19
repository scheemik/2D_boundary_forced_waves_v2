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
# Horizontal wavelength (3 across top boundary)
lam_x = L_x / 3.0
# Oscillation frequency = N_0 * cos(theta), from dispersion relation
omega = 0.7071 # [rad s^-1]
# Angle of beam w.r.t. the horizontal
theta = np.arccos(omega/N_0) # [rad]
# Horizontal wavenumber
k_x    = 2*np.pi/lam_x # [m^-1] k*cos(theta)
# Characteristic wavenumber
k   = k_x*N_0/omega # [m^-1]
# Vertical wavenumber
k_z   = k*np.sin(theta) # [m^-1] k*sin(theta)

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
z_sb     = -1.5

# Background profile parameters
bp_slope = 120
stair_top =  0.0 - 0.2    # m
stair_bot = -L_z + 0.23    # m
bump = 1.3              # N (rad s^-1)
bg_height = N_0         # N (rad s^-1)
