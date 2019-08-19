"""
If the keep (-k) option is set to 1, this script will write out
parameters of the experiment to the relevant log file

Usage:
    write_out_params.py LOC NAME SIM_TYPE NI

Options:
    LOC 	            # -l, 1 if local, 0 if on Niagara
    NAME	            # -n, Name of simulation run
    SIM_TYPE            # -e, Simulation type: (1)Energy flux or (0) reproducing run
    NI		            # Number of inner interfaces

"""

import sys
import numpy as np
# For adding arguments when running
from docopt import docopt

###############################################################################

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    LOC = int(arguments['LOC'])
    LOC = bool(LOC)
    NAME = str(arguments['NAME'])
    SIM_TYPE = int(arguments['SIM_TYPE'])
    NI = int(arguments['NI'])

arctic_params = False

###############################################################################
# Fetch parameters from the correct params file

# Add path to params files
sys.path.insert(0, './_params')
if SIM_TYPE==0:
    # Reproducing results from Ghaemsaidi
    if NI==1:
        import params_repro1
        params = params_repro1
    if NI==2:
        import params_repro2
        params = params_repro2
else:
    # Measuring energy flux
    if arctic_params:
        import params_ef_arctic
        params = params_ef_arctic
    else:
        import params_ef
        params = params_ef

# Domain size
Lx, Lz = float(params.L_x), float(params.L_z) # not including the sponge layer
x_span = params.x_interval # tuple
z_b,z_t= float(params.z_b), float(params.z_t) #(-Lz, 0.0) #(-Lz/2, Lz/2)
x_limits = params.x_limits # tuple
# Characteristic stratification [rad/s]
N0     = float(params.N_0)
# Angle of beam w.r.t. the horizontal
theta  = float(params.theta)
# Wavenumbers
kx, kz = float(params.k_x), float(params.k_z)
# Forcing oscillation frequency
omega  = float(params.omega)
# Oscillation period = 2pi / omega
T      = float(params.T)
# Forcing amplitude modifier
A      = float(params.forcing_amp)
# Forcing amplitude ramp (number of oscillations)
nT     = float(params.nT)
# Horizontal wavelength
lam_x = params.lam_x
if SIM_TYPE==0:
    # Reproducing results from Ghaemsaidi
    # Bounds of the forcing window
    win_left  = params.forcing_left_edge
    win_right = params.forcing_rightedge

# Sponge layer parameters
# Number of grid points in z direction in sponge domain
nz_sp    = int(params.nz_sp)

# Slope of tanh function in slope
sp_slope = float(params.sp_slope)
# Max coefficient for nu at bottom of sponge
max_sp   = float(params.max_sp)

# Bottom of sponge layer
z_sb     = float(params.z_sb)

if LOC:
    import lparams_local
    lparams = lparams_local
else:
    import lparams_Niagara
    lparams = lparams_Niagara

# Number of grid points in each dimension
nx, nz = int(lparams.n_x), int(lparams.n_z)  # doesn't include sponge layer
# Timing of simulation
sim_period_stop  = lparams.sim_period_stop
wall_time_stop = lparams.wall_time_stop
adapt_dt = lparams.adapt_dt
# Calculate stop time
sim_time_stop = sim_period_stop * T # time units (t)

###############################################################################
# Write out to file

# Name of log file
logfile = '_experiments/' + NAME + '/LOG_' + NAME + '.txt'
# Write params to log file
with open(logfile, 'a') as the_file:
    the_file.write('\n')
    the_file.write('--Simulation Parameters--\n')
    the_file.write('\n')
    the_file.write('Sim Runtime (periods):              ' + str(sim_period_stop) + '\n')
    the_file.write('Sim Runtime (seconds):              ' + str(sim_time_stop) + '\n')
    the_file.write('\n')
    the_file.write('Adaptive timestepping:              ' + str(adapt_dt) + '\n')
    the_file.write('\n')
    the_file.write('Horizontal grid points:       n_x = ' + str(nx) + '\n')
    the_file.write('Vertical   grid points:       n_z = ' + str(nz) + '\n')
    the_file.write('\n')
    the_file.write('--Simulation Domain Parameters--\n')
    the_file.write('\n')
    the_file.write('Horizontal extent (m):        L_x = ' + str(Lx) + '\n')
    the_file.write('Vertical   extent (m):        L_z = ' + str(z_t-z_sb) + '\n')
    the_file.write('\n')
    the_file.write('Left   side:                       ' + str(x_span[0]) + '\n')
    the_file.write('Right  side:                        ' + str(x_span[1]) + '\n')
    the_file.write('Top    side:                        ' + str(z_t) + '\n')
    the_file.write('Bottom side:                       ' + str(z_sb) + '\n')
    the_file.write('\n')
    the_file.write('--Measurement Domain Parameters--\n')
    the_file.write('\n')
    the_file.write('Horizontal extent (m):        L_x = ' + str(x_limits[1]-x_limits[0]) + '\n')
    the_file.write('Vertical   extent (m):        L_z = ' + str(Lz) + '\n')
    the_file.write('\n')
    the_file.write('Left   side:                        ' + str(x_limits[0]) + '\n')
    the_file.write('Right  side:                        ' + str(x_limits[1]) + '\n')
    the_file.write('Top    side:                        ' + str(z_t) + '\n')
    the_file.write('Bottom side:                       ' + str(z_b) + '\n')
    the_file.write('\n')
    the_file.write('--Boundary Forcing Parameters--\n')
    the_file.write('\n')
    the_file.write('Horizontal wavenumber:        k_x = ' + str(kx) + '\n')
    the_file.write('Vertical   wavenumber:        k_z = ' + str(kz) + '\n')
    the_file.write('\n')
    the_file.write('Forcing frequency (s^-1):   omega = ' + str(omega) + '\n')
    the_file.write('Forcing angle (rad):        theta = ' + str(theta) + '\n')
    the_file.write('Forcing angle (deg):        theta = ' + str(theta*180/np.pi) + '\n')
    the_file.write('\n')
    the_file.write('Forcing period (s):             T = ' + str(T) + '\n')
    the_file.write('Forcing amplitude:              A = ' + str(A) + '\n')
    the_file.write('\n')
    the_file.write('Horizontal wavelength (m):  lam_x = ' + str(lam_x) + '\n')
    if SIM_TYPE==0:
        # Reproducing results from Ghaemsaidi
        the_file.write('Forcing window left edge:          ' + str(win_left) + '\n')
        the_file.write('Forcing window right edge:          ' + str(win_right) + '\n')
    the_file.write('\n')
    the_file.write('--Sponge Layer Parameters--\n')
    the_file.write('\n')
    the_file.write('Sponge layer grid points:           ' + str(nz_sp) + '\n')
    the_file.write('Sponge layer slope:                 ' + str(sp_slope) + '\n')
    the_file.write('Max viscosity coefficient:          ' + str(max_sp) + '\n')
    the_file.write('Bottom of sponge layer:             ' + str(z_sb) + '\n')
