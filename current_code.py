"""
Dedalus script for simulating internal waves through a vertically stratified
fluid with a density profile similar to a double-diffusive staircase.
Originally for 2D Rayleigh-Benard convection.

Modified by Mikhail Schee, May 2019

Usage:
    current_code.py A1 B2 C3 D4

Arguments:
    A1		Dimensionless number: Euler
    B2		Dimensionless number: Fourier
    C3      Dimensionless number: R = (Ra/Pr)^{-1/2}
    D4      Dimensionless number: Richardson

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
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# For adding arguments when running
from docopt import docopt

# Parameters
aspect_ratio = 4.0
Lx, Lz = (aspect_ratio, 1.)
z_b, z_t = (-Lz/2, Lz/2)
# Placeholders for the 3 dimensionless numbers
A1 = 1.1
B2 = 2.2
C3 = 3.3
D4 = 4.4

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    A1 = float(arguments['A1'])
    print('ND A1=',A1)
    B2 = float(arguments['B2'])
    print('ND B2=',B2)
    C3 = float(arguments['C3'])
    print('ND C3=',C3)
    D4 = float(arguments['D4'])
    print('ND D4=',D4)

###############################################################################

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(z_b, z_t), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

###############################################################################

# Creating a wave-making boudary condition
def wave_maker(x, op=0, L_x=2*np.pi, n=10):
    if (op == 0):
        # a plane wave
        return 1.0
    elif (op ==1):
        # a simple, smooth bump function
        return (((1-np.cos(2*np.pi/L_x * x))/2)**n)

def BoundaryForcing(*args):
    # this function applies its arguments and returns the forcing
    t = args[0].value # this is a scalar; we use .value to get its value
    x = args[1].data # this is an array; we use .data to get its values
    ampl = args[2].value
    freq = args[3].value
    return ampl*wave_maker(x, 0, Lx)*np.cos(t*freq)

def Forcing(*args, domain=domain, F=BoundaryForcing):
    """This function takes arguments *args, a function F, and a domain and
    returns a Dedalus GeneralFunction that can be applied."""
    return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

# now we make it parseable, so the symbol BF can be used in equations
# and the parser will know what it is.
de.operators.parseables['BF'] = Forcing

###############################################################################

# Parameters to determine a specific staircase profile
n_layers = 1
layer_ratio = 1
val_bot = 1.0
val_top = -1.0

# Functions to define an arbitrary staircase profile
def lin_profile(z, z_b, z_t, val_b, val_t):
    # Creates a linear profile from (z_b, val_b) to (z_t, val_t)
    values = 0*z
    slope = (val_t - val_b) / (z_t - z_b)
    values = slope*(z - z_b) + val_b
    return values
def val_ni(n, index, val_b, val_t):
    i = index/2
    # returns value at interface i for n layers (bottom interface=0, bottom layer=1)
    return val_b + (i/n) * (val_t - val_b)
def staircase(z, n, ratio, z_b, z_t, val_b, val_t):
    # initialize array of values to be returned
    values = 0*z
    # find the thickness of the layers and the interfaces
    th_l = (z_t - z_b) / (n + (n-1)/ratio)
    th_i = th_l/ratio
    # z is an array of height values
    # function returns a corresponding array of values (density, for example)
    z_i = z_b
    index = 0 # even is interface, odd is layer
    # Loop from bottom z to top z, alternating layers and interfaces
    while (z_i < z_t):
        # Layer
        if (index%2 == 0):
            index += 1
            z_i += th_l
            val_below = val_ni(n, index-1, val_b, val_t)
            val_above = val_ni(n, index+1, val_b, val_t)
            values[(z>(z_i-th_l))&(z<z_i)] = lin_profile(z[(z>(z_i-th_l))&(z<z_i)], z_i-th_l, z_i, val_below, val_above)
        # Interface
        else:
            index += 1
            z_i += th_i
            values[(z>(z_i-th_i))&(z<z_i)] = val_ni(n, index, val_b, val_t)
    return values

###############################################################################

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
problem.meta['p','b','u','w']['z']['dirichlet'] = True
# Parameters for the dimensionless numbers
problem.parameters['A'] = A1
problem.parameters['B'] = B2
problem.parameters['C'] = C3
problem.parameters['D'] = D4

###############################################################################
# Forcing from the boundary

# Defining wavenumbers in terms of angle with respect to horizontal
theta = np.pi/4
k     = 1.0
kx    = k*np.cos(theta)
kz    = k*np.sin(theta)
# Other parameters
g     = 9.81 # m/s^2
A     = 1.0
N0    = 1.0
rho0  = 1.0
omega = 1.0

# Polarization relation (signs implemented later)
PolRel = {'u': A*(g*omega*kz)/(N0**2*kx),
          'w': A*(g*omega)/(N0**2),
          'b': A*g,
          'p': A*(g*rho0*kz)/(kx**2+kz**2)}

# Creating forcing amplitudes
for fld in ['u', 'w', 'b', 'p']:
    BF = domain.new_field()
    BF.meta['x']['constant'] = True  # means the NCC is constant along x
    BF['g'] = PolRel[fld]
    problem.parameters['BF' + fld] = BF  # pass function in as a parameter.
    del BF

# Parameters for boundary forcing
problem.parameters['kx'] = kx
problem.parameters['kz'] = kz
problem.parameters['omega'] = 1.
# Substitutions for boundary forcing
problem.substitutions['fu'] = "-BFu*sin(kx*x + kz*z + omega*t)"
problem.substitutions['fw'] = " BFw*sin(kx*x + kz*z + omega*t)"
problem.substitutions['fb'] = " BFb*cos(kx*x + kz*z + omega*t)"
problem.substitutions['fp'] = "-BFp*sin(kx*x + kz*z + omega*t)"

###############################################################################

# Background Profile (bgpf) as an NCC
bgpf = domain.new_field()
bgpf.meta['x']['constant'] = True  # means the NCC is constant along x
# Store profile in an array so it can be used for initial conditions later
bgpf_array = staircase(z, n_layers, layer_ratio, z_b, z_t, val_bot, val_top)
bgpf['g'] = bgpf_array
problem.parameters['bgpf'] = bgpf  # pass function in as a parameter
del bgpf

###############################################################################

# Non-Dimensionalized Equations
#problem.substitutions['v*del(u)']
#   Mass conservation equation
problem.add_equation("dx(u) + wz = 0")
#   Equation of state (in terms of buoyancy)
problem.add_equation("dt(b) - B*(dx(dx(b)) + dz(bz))                = -(u*dx(b) + w*bz)")
#   Horizontal momentum equation
problem.add_equation("dt(u) - C*(dx(dx(u)) + dz(uz)) + A*dx(p)      = -(u*dx(u) + w*uz)")
#   Vertical momentum equation
problem.add_equation("dt(w) - C*(dx(dx(w)) + dz(wz)) + A*dz(p) - b  = D*bgpf -(u*dx(w) + w*wz)")

# Required for differential equation solving in Chebyshev
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")

###############################################################################

# Boundary contitions
#	Using Fourier basis for x automatically enforces periodic bc's
#   Left is bottom, right is top
# Solid top/bottom boundaries
problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = right(fu)") #0")
# Free top/bottom boundaries
#problem.add_bc("left(uz) = 0")
#problem.add_bc("right(uz) = 0")
# No-slip top/bottom boundaries?
problem.add_bc("left(w) = 0", condition="(nx != 0)") # redunant in constant mode (nx==0)
problem.add_bc("right(w) = right(fw)") #0")
# Buoyancy = zero at top/bottom
problem.add_bc("left(b) = 0")
problem.add_bc("right(b) = right(fb)") #0")
# Sets gauge pressure to zero in the constant mode
problem.add_bc("left(p) = 0", condition="(nx == 0)") # required because of above redundancy

###############################################################################

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
b = solver.state['b']
bz = solver.state['bz']
'''
# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)
'''
# Buoyancy initial conditions
b['g'] = bgpf_array #C3 * pert
b.differentiate('z', out=bz)

###############################################################################

# Initial timestep
dt = 0.125

# Integration parameters
solver.stop_sim_time = 25
solver.stop_wall_time = 30 * 60.
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50)
snapshots.add_system(solver.state)

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
flow.add_property("sqrt(u*u + w*w) / A", name='Re') # this is no longer correct. change to Rossby number?

###############################################################################

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
