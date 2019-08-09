# Parameters for measuring energy flux for the arctic ocean


import numpy as np

# Domain size
L_x = 5.0 # m
L_z = 10. # m, not including the sponge layer
x_interval = (0.0, L_x)
z_b, z_t = -L_z, 0.0
# Display limits for plot
x_limits=[0.0, L_x]

# Aspect ratio of display domain
aspect_ratio = 2.0

# Boundary forcing parameters
# Characteristic stratification
N_0 = 0.005 # [rad/s]
# Angle of beam w.r.t. the horizontal
theta = np.pi/4
# Horizontal wavelength
lam_x = L_x
# Horizontal wavenumber
k_x    = 2*np.pi/lam_x
# Vertical wavenumber = 2*pi/lam_z, or from trig:
k_z    = k_x * np.tan(theta)
# Oscillation frequency = N_0 * cos(theta), from dispersion relation
omega  = N_0 * np.cos(theta) # [s^-1]
# Oscillation period = 2pi / omega
T = 2*np.pi / omega
# Forcing amplitude modifier
forcing_amp = 1.0e-4
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
z_sb     = -L_z - 2.0

# Background profile parameters
bp_slope = 100
st_buffer = L_z/5.0   # m
bump = 0.02           # N (rad s^-1)
bg_height = N_0       # N (rad s^-1)
