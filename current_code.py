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

This script uses a Fourier basis in the x direction with periodic boundary conditions.

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
# Switchboard

# Reproducing run, or measuring energy flux?
reproducing_run = False
# Save just the relevant snapshots, or all?
save_all_snapshots = True

# Optional outputs (will not plot if run remotely)
plot_z_basis = False
plot_SL = False
plot_BP = True
print_params = True

# Options for simulation
use_sponge_layer = True
set_N_const = True

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

###############################################################################
# Fetch parameters from the correct params file

# Add path to params files
sys.path.insert(0, './_params')
if LOC:
    if reproducing_run:
        print('Reproducing results from Ghaemsaidi')
        import params_repro
        params = params_repro
    else:
        print('Measuring energy flux')
        import params_ef
        params = params_ef
elif LOC == False:
    import params_Niagara
    params = params_Niagara

# Aspect ratio of domain of interest
AR = float(params.aspect_ratio)
# Number of layers in background profile
NL = int(params.n_layers)
# Number of grid points in each dimension
nx, nz = int(params.n_x), int(params.n_z)  # doesn't include sponge layer
# Domain size
Lx, Lz = float(params.L_x), float(params.L_z) # not including the sponge layer
x_span = params.x_interval # tuple
z_b, z_t = float(params.z_b), float(params.z_t) #(-Lz, 0.0) #(-Lz/2, Lz/2)
# Angle of beam w.r.t. the horizontal
theta   = float(params.theta)
# Wavenumbers
kx, kz = float(params.k_x), float(params.k_z)
# Forcing amplitude modifier
A       = float(params.forcing_amp)
if (rank==0 and print_params):
    print('n_x=',nx)
    print('n_z=',nz)
sim_time_stop  = params.sim_time_stop
wall_time_stop = params.wall_time_stop
adapt_dt = params.adapt_dt

###############################################################################
# Physical parameters
nu    = NU          # [m^2/s]   Viscosity (momentum diffusivity)
kappa = KA          # [m^2/s]   Thermal diffusivity
Pr    = NU/KA       # [nondim]  Prandtl number, nu/kappa = 7 for water
if (rank==0 and print_params): print('Prandtl number =',Pr)
#rho_0 = R0          # [kg/m^3]  Characteristic density -> now wrapped into pressure
N_0   = N0          # [rad/s]   Characteristic stratification
g     = 9.81        # [m/s^2]   Acceleration due to gravity
omega = N_0 * np.cos(theta) # [s^-1], from dispersion relation

###############################################################################
# Parameters to set a sponge layer at the bottom
nz_sp = 40          # number of grid points in z direction in sponge domain
sp_slope = -20.     # slope of tanh function in slope
max_sp   =  50.     # max coefficient for nu at bottom of sponge
H_sl     =  0.5     # height of sponge layer = 2 * H_sl * Lz
z_sb     = z_b-2*H_sl*Lz      # bottom of sponge layer

###############################################################################

# Create bases and domain
x_basis  = de.Fourier('x', nx, interval=x_span, dealias=3/2)
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

if reproducing_run:
    # Windowing function (multiplying tanh's)
    # Slope of tanh for forcing window
    f_slope = float(params.forcing_slope)
    # Bounds of the forcing window
    fl_edge, fr_edge = float(params.forcing_left_edge), float(params.forcing_rightedge)
    problem.parameters['slope'] = f_slope
    problem.parameters['left_edge'] = fl_edge
    problem.parameters['right_edge'] = fr_edge
    problem.substitutions['window'] = "(1/2)*(tanh(slope*(x-left_edge))+1)*(1/2)*(tanh(slope*(-x+right_edge))+1)"
else:
    problem.substitutions['window'] = "1"

# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['fu'] = "-BFu*sin(kx*x + kz*z - omega*t)*window"
problem.substitutions['fw'] = " BFw*sin(kx*x + kz*z - omega*t)*window"
problem.substitutions['fb'] = "-BFb*cos(kx*x + kz*z - omega*t)*window"
#problem.substitutions['fp'] = "-BFp*sin(kx*x + kz*z - omega*t)*window"

###############################################################################
# Plotting function for sponge layer, background profile, etc.
def test_plot(vert, hori, plt_title, x_label, y_label, y_lims):
    with plt.rc_context({'axes.edgecolor':'white', 'text.color':'white', 'axes.labelcolor':'white', 'xtick.color':'white', 'ytick.color':'white', 'figure.facecolor':'black'}):
        fg, ax = plt.subplots(1,1)
        ax.set_title(plt_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(y_lims)
        ax.plot(hori, vert, 'k-')
        plt.grid(True)
        plt.show()
###############################################################################

# Sponge Layer (SL) as an NCC
#   Check state of `use_sponge_layer` above
SL = domain.new_field()
SL.meta['x']['constant'] = True  # means the NCC is constant along x
# Import the sponge layer function from the sponge layer script
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
    plt_title = 'Sponge Profile'
    x_label = r'viscosity coefficient'
    y_label = r'depth ($z$)'
    y_lims  = [z_sb,z_t]
    test_plot(vert, hori, plt_title, x_label, y_label, y_lims)

###############################################################################

# Background Profile (BP) as an NCC
print_arrays = False
BP = domain.new_field()
BP.meta['x']['constant'] = True  # means the NCC is constant along x
# Construct profile in an array so it can be used later
if set_N_const:
    BP_array = z*0 + 1.0
else: # Construct a staircase profile
    n_layers = NL
    # Import the staircase function from the background profile script
    sys.path.insert(0, './_background_profile')
    if reproducing_run:
        slope = float(params.profile_slope)#*(n_layers+1)
        N_1 = float(params.N_1)                  # Stratification value above staircase
        N_2 = float(params.N_2)                  # Stratification value below staircase
        if n_layers == 1:                        # Top of staircase (not domain)
            st_bot = float(params.stair_bot_1)
        elif n_layers == 2:
            st_bot = float(params.stair_bot_2)
        else:
            print("NL must be 1 or 2 for reproduction run")
        st_top = float(params.stair_top)         # Bottom of staircase (not domian)
        from Foran_profile import Foran_profile
        BP_array = Foran_profile(z, n_layers-1, st_bot, st_top, slope, N_1, N_2)
    else:
        slope = 100
        st_buffer = 0.1
        bump = 1.3
        st_bot = z_b + st_buffer
        st_top = z_t - st_buffer
        from background_profile import N2_profile
        BP_array = N2_profile(z, n_layers, st_bot, st_top, slope, bump)
BP['g'] = BP_array
problem.parameters['BP'] = BP  # pass function in as a parameter
del BP

# Save background N profile to file so plotting function can read in
vert = np.array(z[0])
hori = np.array(BP_array[0])
filename = '_background_profile/current_N_'
np.save(filename+'x', hori)
np.save(filename+'y', vert)
if (rank==0 and print_arrays):
    print('hori')
    print(hori)
    print('vert')
    print(vert)

# Plots the background profile
if (plot_BP and rank == 0 and LOC):
    plt_title = 'Background Profile'
    x_label = r'frequency ($N^2$)'
    y_label = r'depth ($z$)'
    y_lims  = [z_b,z_t]
    test_plot(vert, hori, plt_title, x_label, y_label, y_lims)

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
if (save_all_snapshots or reproducing_run):
    snapshots_path = 'snapshots'
    snapshots = solver.evaluator.add_file_handler(snapshots_path, sim_dt=0.25, max_writes=50)
    snapshots.add_system(solver.state)

# CFL - adapts time step as the code runs depending on stiffness
#       implemented in 'while solver.ok' loop
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=10)
# Some other criterion
flow.add_property("dx(u)/omega", name='Lin_Criterion')

###############################################################################
# Measuring "energy flux" through a horizontal boundary at some z

# Adding a new file handler
if (save_all_snapshots or reproducing_run==False):
    ef_snapshots_path = 'ef_snapshots'
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
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
