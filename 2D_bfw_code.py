"""
Dedalus script for 2D Rayleigh-Benard convection.

Usage:
    rb_with_S.py RA PR

Arguments:
    RA		Rayleigh number
    PR		Prandtl number

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

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
#   for free boundaries, I expect the rolls to have an
#   aspect ratio of sqrt(2). See Lautrup fig 30.3
aspect_ratio = np.sqrt(2.)
#   make the horizontal axis fit 4 rolls side by side
Lx, Lz = (4.*aspect_ratio, 1.)
Prandtl = 1.
Rayleigh = 1e6

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    Rayleigh = float(arguments.get('RA'))
    print('Rayleigh number = ', Rayleigh)
    Prandtl = float(arguments.get('PR'))
    print('Prandtl number = ', Prandtl)

# Create bases and domain
#x_basis = de.Fourier('x', 256, interval=(0, Lx), dealias=3/2)
x_basis = de.Fourier('x', 256, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis = de.Chebyshev('z', 64, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','r','u','w','rz','uz','wz'])
problem.meta['p','r','u','w']['z']['dirichlet'] = True
problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
problem.parameters['F'] = F = 1
# Mass conservation equation
problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(r) - F*(dx(dx(r)) + dz(rz)) = -(u*dx(r) + w*rz)")
problem.add_equation("dt(u) + dx(p)       = -(u*dx(u) + w*uz)")
problem.add_equation("dt(w) + dz(p) - P*r = -(u*dx(w) + w*wz)")
# Definitions for easier derivative syntax
problem.add_equation("rz - dz(r) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
# Boundary contitions
problem.add_bc("left(r) = 0")
# Solid boundaries
#problem.add_bc("left(u) = 0")
# Free boundaries
#problem.add_bc("left(uz) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(r) = 0")
# Solid boundaries
#problem.add_bc("right(u) = 0")
# Free boundaries
problem.add_bc("right(uz) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

# Build solver
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions
x = domain.grid(0)
z = domain.grid(1)
r = solver.state['r']
rz = solver.state['rz']

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)
r['g'] = F * pert
r.differentiate('z', out=rz)

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
flow.add_property("sqrt(u*u + w*w) / R", name='Re')

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
