"""
If the keep (-k) option is set to 1, this script will write out
parameters of the experiment to the relevant log file

Usage:
    write_out_params.py LOC NAME SIM_TYPE

Options:
    LOC 	            # -l, 1 if local, 0 if on Niagara
    NAME	            # -n, Name of simulation run
    SIM_TYPE            # -e, Simulation type: (1)Energy flux or (0) reproducing run

"""

import sys
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

###############################################################################
# Fetch parameters from the correct params file

# Add path to params files
sys.path.insert(0, './_params')
if SIM_TYPE==0:
    # Reproducing results from Ghaemsaidi
    import params_repro
    params = params_repro
else:
    # Measuring energy flux
    import params_ef
    params = params_ef

# Simulation time stop in periods
sim_period_stop = params.sim_period_stop
# Simulation time stop in seconds
sim_time_stop  = params.sim_time_stop
# Number of grid points in each dimension
nx, nz = int(params.n_x), int(params.n_z)  # doesn't include sponge layer
# Domain size
Lx, Lz = float(params.L_x), float(params.L_z) # not including the sponge layer
# Angle of beam w.r.t. the horizontal
theta   = float(params.theta)
# Wavenumbers
kx, kz = float(params.k_x), float(params.k_z)
# Forcing oscillation frequency
omega  = float(params.omega)
# Forcing oscillation period
T      = float(params.T)
# Forcing amplitude modifier
A       = float(params.forcing_amp)

###############################################################################
# Write out to file

# Name of log file
logfile = '_experiments/' + NAME + '/LOG_' + NAME + '.txt'
# Write params to log file
with open(logfile, 'a') as the_file:
    the_file.write('\n')
    the_file.write('--Experiment Parameters:\n')
    the_file.write('\n')
    the_file.write('Sim Runtime (periods):              ' + str(sim_period_stop) + '\n')
    the_file.write('Sim Runtime (seconds):              ' + str(sim_time_stop) + '\n')
    the_file.write('\n')
    the_file.write('Horizontal grid points:       n_x = ' + str(nx) + '\n')
    the_file.write('Vertical grid points:         n_z = ' + str(nz) + '\n')
    the_file.write('\n')
    the_file.write('Horizontal extent (m):        L_x = ' + str(Lx) + '\n')
    the_file.write('Vertical extent (m):          L_z = ' + str(Lz) + '\n')
    the_file.write('\n')
    the_file.write('Forcing frequency (s^-1):   omega = ' + str(omega) + '\n')
    the_file.write('Forcing period (s):             T = ' + str(T) + '\n')
    the_file.write('Forcing angle (rad):        theta = ' + str(theta) + '\n')
    the_file.write('Forcing amplitude:              A = ' + str(A) + '\n')
    the_file.write('\n')
    the_file.write('Horizontal wavenumber:        k_x = ' + str(kx) + '\n')
    the_file.write('Vertical wavenumber:          k_z = ' + str(kz) + '\n')
