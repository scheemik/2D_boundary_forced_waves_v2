"""
Dedalus script for simulating internal waves through a vertically stratified
fluid with a density profile similar to a double-diffusive staircase.
Originally for 2D Rayleigh-Benard convection.

Modified by Mikhail Schee, June 2019

Usage:
    current_code.py LOC AR NU KA N0 NL

Arguments:
    LOC     # 1 if local, 0 if on Niagara
    AR		# [nondim]  Aspect ratio of domain
    NU		# [m^2/s]   Viscosity (momentum diffusivity)
    KA		# [m^2/s]   Thermal diffusivity
    N0		# [rad/s]   Characteristic stratification
    NL		# [nondim]	Number of inner interfaces

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
import matplotlib.pyplot as plt
import time
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
#print("thread %d of %d" % (comm.Get_rank(), comm.Get_size()))
size = comm.Get_size()
rank = comm.Get_rank()

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

# For adding arguments when running
from docopt import docopt

###############################################################################
###############################################################################
# Switchboard

# Optional outputs (will not plot if run remotely)
plot_z_basis = False
plot_SL = False
plot_BP = False
print_params = True

# Options for simulation
use_sponge_layer = True
set_N_const = False

###############################################################################
###############################################################################

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    LOC = int(arguments['LOC'])
    LOC = bool(LOC)
    AR = float(arguments['AR'])
    NU = float(arguments['NU'])
    KA = float(arguments['KA'])
    #R0 = float(arguments['R0'])
    N0 = float(arguments['N0'])
    NL = int(arguments['NL'])
    if (rank == 0 and print_params):
        print('LOC=',LOC)
        print('AR =',AR)
        print('NU =',NU)
        print('KA =',KA)
        #print('R0=',R0)
        print('N0 =',N0)
        print('NL =',NL)

# Parameters
aspect_ratio = AR
Lx, Lz = (aspect_ratio, 1.)
z_b, z_t = (-Lz/2, Lz/2)
# Placeholders for params
nx = 40*2       # number of grid points in x direction
nz = 40*1       # number of grid points in z direction in main domain

###############################################################################
# Fetch parameters from the correct params file

# Add correct path to params files
sys.path.insert(0, './_params')
if LOC:
    import params_local
    params = params_local
elif LOC == False:
    import params_Niagara
    params = params_Niagara

nx = int(params.n_x)
nz = int(params.n_z)
if (rank==0 and print_params):
    print('n_x=',nx)
    print('n_z=',nz)
sim_time_stop  = params.sim_time_stop
wall_time_stop = params.wall_time_stop
adapt_dt = params.adapt_dt

###############################################################################
# Setting parameters

# Physical parameters
nu    = NU          # [m^2/s]   Viscosity (momentum diffusivity)
kappa = KA          # [m^2/s]   Thermal diffusivity
Pr    = NU/KA       # [nondim]  Prandtl number, nu/kappa = 7 for water
if (rank==0): print('Prandtl number =',Pr)
#rho_0 = R0          # [kg/m^3]  Characteristic density -> now wrapped into pressure
N_0   = N0          # [rad/s]   Characteristic stratification
g     = 9.81        # [m/s^2]   Acceleration due to gravity

# Boundary forcing parameters
# Bounds of the forcing window
fl_edge = -3.0*Lx/12.0
fr_edge =  -Lx/12.0
# Angle of beam w.r.t. the horizontal
theta = np.pi/4
# Horizontal wavelength
lam_x = fr_edge - fl_edge
# Horizontal wavenumber
kx    = 2*np.pi/lam_x
# Vertical wavenumber = 2*pi/lam_z, or from trig:
kz    = kx * np.tan(theta)
# Other parameters
A     = 3.0e-4
omega = N_0 * np.cos(theta) # [s^-1], from dispersion relation

###############################################################################

# Parameters to set a sponge layer at the bottom

nz_sp = 40          # number of grid points in z direction in sponge domain
sp_slope = -10.     # slope of tanh function in slope
max_sp   =  50.     # max coefficient for nu at bottom of sponge
H_sl     =  0.5     # height of sponge layer = 2 * H_sl * Lz
z_sb     = z_b-2*H_sl*Lz      # bottom of sponge layer

###############################################################################

# Create bases and domain
x_basis  = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
if use_sponge_layer:
    z_main   = de.Chebyshev('zm', nz, interval=(z_b, z_t), dealias=3/2)
    z_sponge = de.Chebyshev('zs', nz_sp, interval=(z_sb, z_b), dealias=3/2)
    z_basis  = de.Compound('z', (z_sponge, z_main))
else:
    z_basis = de.Chebyshev('z', nz, interval=(z_b, z_t), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
# Initial conditions
x = domain.grid(0)
z = domain.grid(1)

# Making a plot of the grid spacing in the z direction
if (plot_z_basis and rank==0 and LOC):
    # scale should match dealias
    grid_spacing = z_basis.grid(scale=3/2)
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        plt.figure(figsize=(10, 1))
        plt.plot(grid_spacing, np.zeros_like(grid_spacing)+1, '.', markersize=2)
        # Makes a dividing line between the sponge and the main domain
        plt.plot((z_b, z_b), (0, 2), 'k--')
        plt.title('Vertical grid spacing')
        plt.xlabel(r'depth ($z$)')
        plt.ylim([0, 2])
        plt.gca().yaxis.set_ticks([])
        plt.show()

###############################################################################

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz'])
# From Nico: all variables are dirchlet by default, so only need to
#   specify those that are not dirchlet (variables w/o top & bottom bc's)
problem.meta['p','bz','uz','wz']['z']['dirichlet'] = False
# Parameters for the equations of motion
problem.parameters['NU'] = nu
problem.parameters['KA'] = kappa
#problem.parameters['R0'] = rho_0
problem.parameters['N0'] = N_0

###############################################################################
# Forcing from the boundary

# Polarization relation from Cushman-Roisin and Beckers eq (13.7)
#   (signs implemented later)
PolRel = {'u': A*(g*omega*kz)/(N_0**2*kx),
          'w': A*(g*omega)/(N_0**2),
          'b': A*g}#,
          #'p': A*(g*rho_0*kz)/(kx**2+kz**2)} # relation for p not used

# Creating forcing amplitudes
for fld in ['u', 'w', 'b']:#, 'p']:
    BF = domain.new_field()
    BF.meta['x']['constant'] = True  # means the NCC is constant along x
    BF['g'] = PolRel[fld]
    problem.parameters['BF' + fld] = BF  # pass function in as a parameter.
    del BF

# Parameters for boundary forcing
problem.parameters['kx'] = kx
problem.parameters['kz'] = kz
problem.parameters['omega'] = omega
problem.parameters['grav'] = g # can't use 'g' because Dedalus already uses that for grid

# Windowing function (multiplying tanh's)
problem.parameters['slope'] = 10
problem.parameters['left_edge'] = fl_edge
problem.parameters['right_edge'] = fr_edge
problem.substitutions['window'] = "(1/2)*(tanh(slope*(x-left_edge))+1)*(1/2)*(tanh(slope*(-x+right_edge))+1)"

# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['fu'] = "-BFu*sin(kx*x + kz*z - omega*t)*window"
problem.substitutions['fw'] = " BFw*sin(kx*x + kz*z - omega*t)*window"
problem.substitutions['fb'] = "-BFb*cos(kx*x + kz*z - omega*t)*window"
#problem.substitutions['fp'] = "-BFp*sin(kx*x + kz*z - omega*t)*window"

###############################################################################

# Sponge Layer (SL) as an NCC
#   Check state of `use_sponge_layer` above
SL = domain.new_field()
SL.meta['x']['constant'] = True  # means the NCC is constant along x
# Import the sponge layer function from the sponge layer script
#import sys
sys.path.insert(0, './_sponge_layer')
from sponge_layer import sponge_profile
# Store profile in an array so it can be used later
if use_sponge_layer:
    SL_array = sponge_profile(z, z_sb, z_t, sp_slope, max_sp, H_sl)
else:
    SL_array = z*0.0 + 1.0
SL['g'] = SL_array
problem.parameters['SL'] = SL  # pass function in as a parameter
#   Multiply nu by SL in the equations of motion
del SL

# Plots the sponge layer coefficient profile
if (plot_SL and rank == 0 and LOC):
    vert = np.array(z[0])
    hori = np.array(SL_array[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax = plt.subplots(1,1)
        ax.set_title('Sponge Profile')
        ax.set_xlabel(r'viscosity coefficient')
        ax.set_ylabel(r'depth ($z$)')
        ax.set_ylim([z_sb,z_t])
        ax.plot(hori, vert, '-')
        plt.grid(True)
        plt.show()

###############################################################################

# Parameters to determine a specific staircase profile
n_layers = NL - 1
slope = 100.0*(n_layers+1)
N_1 = 0.95                  # The stratification value above the staircase
N_2 = 1.24                  # The stratification value below the staircase
z_bot = -0.1                # Top of staircase (not domain)
z_top =  0.1                # Bottom of staircase (not domian)

###############################################################################

# Background Profile (BP) as an NCC
BP = domain.new_field()
BP.meta['x']['constant'] = True  # means the NCC is constant along x
# Import the staircase function from the background profile script
#import sys
sys.path.insert(0, './_background_profile')
from Foran_profile import Foran_profile
# Store profile in an array so it can be used later
if set_N_const:
    BP_array = z*0 + 1.0
else:
    BP_array = Foran_profile(z, n_layers, z_bot, z_top, slope, N_1, N_2)
BP['g'] = BP_array
problem.parameters['BP'] = BP  # pass function in as a parameter
del BP

# Plots the background profile
if (plot_BP and rank == 0 and LOC):
    vert = np.array(z[0])
    hori = np.array(BP_array[0])
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax = plt.subplots(1,1)
        ax.set_title('Background Profile')
        ax.set_xlabel(r'frequency ($N^2$)')
        ax.set_ylabel(r'depth ($z$)')
        ax.set_ylim([z_sb,z_t])
        ax.plot(hori, vert, '-')
        plt.grid(True)
        # Uncomment to save figure
        #plt.savefig('2019_07_10-N2_profile_w_SL.png', facecolor=fg.get_facecolor(), transparent=True)
        plt.show()

###############################################################################

# Equations of motion (non-linear terms on RHS)
#   Mass conservation equation
problem.add_equation("dx(u) + wz = 0")
#   Equation of state (in terms of buoyancy)
problem.add_equation("dt(b) - KA*(dx(dx(b)) + dz(bz))"
                    + "= -((N0*BP)**2)*w - (u*dx(b) + w*bz)")
#   Horizontal momentum equation
problem.add_equation("dt(u) - SL*NU*(dx(dx(u)) + dz(uz)) + dx(p)" #/R0"
                    + "= - (u*dx(u) + w*uz)")
#   Vertical momentum equation
problem.add_equation("dt(w) - SL*NU*(dx(dx(w)) + dz(wz)) + dz(p) - b" #/R0"
                    + "= - (u*dx(w) + w*wz)")

# Required for differential equation solving in Chebyshev dimension
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

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-3 * noise * (zt - z) * (z - zb)

# Buoyancy initial conditions
b['g'] = pert * 0.0
print('rank',rank,'has shape =',np.shape(b['g']))
b.differentiate('z', out=bz)

###############################################################################

# Initial timestep
dt = 0.125

# Integration parameters
solver.stop_sim_time = sim_time_stop
solver.stop_wall_time = wall_time_stop * 60.
solver.stop_iteration = np.inf

# Analysis
if True:
    snapshots_path = 'snapshots'
else:
    snapshots_path = '/scratch/n/ngrisoua/mschee/Dedalus/Vanilla_Dedalus/2D_boundary_forced_waves_v2/snapshots'
snapshots = solver.evaluator.add_file_handler(snapshots_path, sim_dt=0.25, max_writes=50)
snapshots.add_system(solver.state)

# CFL - adapts time step as the code runs depending on stiffness
#       implemented in 'while solver.ok' loop
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
# Reduced Richardson number
#flow.add_property("(4*((N0*BP)**2 + bz) - (uz**2))/N0**2", name='Ri_red')
# Some other criterion
flow.add_property("dx(u)/omega", name='Lin_Criterion')

###############################################################################
# Measuring "energy flux" through a horizontal boundary at some z

# Adding a new file handler
if True:
    ef_snapshots_path = 'ef_snapshots'
else:
    ef_snapshots_path = '/scratch/n/ngrisoua/mschee/Dedalus/Vanilla_Dedalus/2D_boundary_forced_waves_v2/ef_snapshots'
ef_snapshots = solver.evaluator.add_file_handler(ef_snapshots_path, sim_dt=0.25, max_writes=100)
# Adding a task to integrate energy flux across x for values of z
ef_snapshots.add_task("integ(0.5*(w*u**2 + w**3) + p*w - NU*(u*uz + w*wz), 'x')", layout='g', name='<ef>')
#ef_snapshots.add_task("integ(0.5*(w*u**2 + w**3) + grav*z*w, 'x')", layout='g', name='<ef>')

###############################################################################

# Adaptive time stepping turned on/off in params file

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        if (adapt_dt == True):
            dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max linear criterion = {0:f}'.format(flow.max('Lin_Criterion')))
            if np.isnan(flow.max('Lin_Criterion')):
                raise NameError('Code blew up it seems')
            '''
            logger.info('Min reduced Ri = {0:f}'.format(flow.min('Ri_red')))
            if np.isnan(flow.min('Ri_red')):
                raise NameError('Code blew up it seems')
            '''


except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
