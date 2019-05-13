"""
Dedalus script for 2D Rayleigh-Benard convection.

Modified by Mikhail Schee, May 2019

Usage:
    rb_with_S.py A1 B2 C3

Arguments:
    A1		Dimensionless number: Euler
    B2		Dimensionless number: Fourier
    C3      Dimensionless number: Froude

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
# Placeholders for the 3 dimensionless numbers
A1 = 1.1
B2 = 2.2
C3 = 3.3

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    A1 = float(arguments['A1'])
    print('ND A1=',A1)
    B2 = float(arguments['B2'])
    print('ND B2=',B2)
    C3 = float(arguments['C3'])
    print('ND C3=',C3)

# Create bases and domain
x_basis = de.Fourier('x', 256, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
'''
# Defining the background density profile
rho_min = 1.1
rho_max = 1.2
def density_profile(*args):
	# this function applies its arguments and returns the profile
	z = args[0].value
	rho_min = args[1].value
	rho_max = args[2].value
	return rho_bar(z, rho_min, rho_max)

def rho_bar(z, rho_min, rho_max):
	# defines the background density to be a linear gradient
	return z*(rho_max-rho_min)/Lz + (rho_max + rho_min)/2.0

def profiling(*args, domain=domain, F=density_profile):
	return de.operators.GeneralFunction(domain, layout='g', func=F, args=args)

# Make the density profile parseable
de.operators.parseables['DP'] = profiling
'''

'''
# Adding a non-constant coefficient (ncc)
ncc = domain.new_field(name='c')
ncc['g'] = z**2
ncc.meta['x', 'y']['constant'] = True
problem.paramters['c'] = ncc
'''

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz'])
#problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
problem.meta['p','b','u','w']['z']['dirichlet'] = True
problem.parameters['A'] = A1 #(Rayleigh * Prandtl)**(-1/2)
problem.parameters['B'] = B2 #(Rayleigh / Prandtl)**(-1/2)
problem.parameters['C'] = C3 #F = 1
'''
problem.parameters['rho_min'] = rho_min
problem.parameters['rho_max'] = rho_max
'''

# Non-Dimensionalized Equations, no viscosity
#   Mass conservation equation
problem.add_equation("dx(u) + dz(w) = 0") #wz = 0")
#   Energy equation (in terms of buoyancy)
problem.add_equation("dt(b) - B*(dx(dx(b)) + dz(bz)) = -(u*dx(b) + w*bz)")
#   Horizontal momentum equation
problem.add_equation("dt(u) + A*dx(p) = -(u*dx(u) + w*dz(u))")
#   Vertical momentum equation
problem.add_equation("dt(w) + A*dz(p) - b = -(u*dx(w) + w*dz(w))")#wz)")
#problem.add_equation("dt(w) + A*dz(p) - b - (-1.0/(C**2))*DP(z,rho_min,rho_max)= -(u*dx(w) + w*wz)") # can't have independent variables (x,z) in the eqs

# Required for differential equation solving in Chebyshev
problem.add_equation("bz - dz(b) = 0")
#problem.add_equation("uz - dz(u) = 0") # redundant
#problem.add_equation("wz - dz(w) = 0") # redundant

# Boundary contitions
#	Using Fourier basis for x automatically enforces periodic bc's
#   Left is bottom, right is top
# Solid top/bottom boundaries
#problem.add_bc("left(u) = 0")
#problem.add_bc("right(u) = 0")
# Free top/bottom boundaries
#problem.add_bc("left(uz) = 0")
#problem.add_bc("right(uz) = 0")
# No-slip top/bottom boundaries?
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
# Buoyancy = zero at top/bottom
problem.add_bc("left(b) = 0")
problem.add_bc("right(b) = 0")
# Sets gauge pressure to zero in the constant mode
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]
'''
# Defining the background density gradient
def bg_density(z, rho_min, rho_max):
	# defines the background density to be a linear gradient
	return z*(rho_max-rho_min)/Lz + (rho_max + rho_min)/2.0
'''
# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)
b['g'] = C3 * pert
#b['g'] = (-1.0/(C3**2)) * bg_density(z, rho_min, rho_max)
b.differentiate('z', out=bz)

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
flow.add_property("sqrt(u*u + w*w) / A", name='Re') # this is no longer correct

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
