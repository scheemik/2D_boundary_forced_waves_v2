"""
Dedalus script for simulating internal waves through a vertically stratified
fluid with a density profile similar to a double-diffusive staircase.
Originally for 2D Rayleigh-Benard convection.

Modified by Mikhail Schee, June 2019

Usage:
    current_code.py LOC SIM_TYPE NU KA NI TEST_P

Arguments:
    LOC         # 1 if local, 0 if on Niagara
    SIM_TYPE    # -e, Simulation type: (1)Energy flux or (0) reproducing run
    NU		    # [m^2/s]   Viscosity (momentum diffusivity)
    KA		    # [m^2/s]   Thermal diffusivity
    NI		    # [nondim]	Number of inner interfaces
    TEST_P	    # Test parameter

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
import matplotlib
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
from dedalus.extras import plot_tools

import logging
logger = logging.getLogger(__name__)

# For adding arguments when running
from docopt import docopt

###############################################################################
# Switchboard

# Save just the relevant snapshots, or all?
save_all_snapshots = True

# Optional outputs (will not plot if run remotely)
plot_z_basis = False
plot_SL = False
plot_BP = False
print_params = True

# Not actually sponge layer, just extends the domain downwards
use_sponge_layer = True

# Only for EF runs
arctic_params = False

###############################################################################

# Read in parameters from docopt
if __name__ == '__main__':
    arguments = docopt(__doc__)
    LOC = int(arguments['LOC'])
    LOC = bool(LOC)
    SIM_TYPE = int(arguments['SIM_TYPE'])
    NU = float(arguments['NU'])
    KA = float(arguments['KA'])
    NI = int(arguments['NI'])
    TEST_P = float(arguments['TEST_P'])
    if (rank == 0 and print_params):
        print('LOC =',LOC)
        print('SIM_TYPE =',SIM_TYPE)
        print('NU =',NU)
        print('KA =',KA)
        print('NI =',NI)
        print('TEST_P =',TEST_P)

###############################################################################
# Fetch parameters from the correct params files

# Add path to params files
sys.path.insert(0, './_params')
if SIM_TYPE==0:
    if rank==0:
        print('Reproducing results from Ghaemsaidi')
    if NI==1:
        import params_repro1
        params = params_repro1
    if NI==2:
        import params_repro2
        params = params_repro2
else:
    if rank==0:
        print('Measuring energy flux')
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
if (rank==0 and print_params):
    print('n_x =',nx)
    print('n_z =',nz)
    print('simulation stop time (s) =', sim_time_stop)
    print('')

###############################################################################
# Physical parameters
nu    = NU          # [m^2/s]   Viscosity (momentum diffusivity)
kappa = KA          # [m^2/s]   Thermal diffusivity
Pr    = NU/KA       # [nondim]  Prandtl number, nu/kappa = 7 for water
if (rank==0 and print_params): print('Prandtl number =',Pr)
g     = 9.81        # [m/s^2]   Acceleration due to gravity

###############################################################################
# Parameters to set a sponge layer at the bottom

# Number of grid points in z direction in sponge domain
nz_sp    = params.nz_sp

# Slope of tanh function in slope
sp_slope = params.sp_slope
# Max coefficient for nu at bottom of sponge
max_sp   = params.max_sp

# Bottom of sponge layer
z_sb     = params.z_sb

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
problem.parameters['N0'] = N0

###############################################################################
# Forcing from the boundary

# Polarization relation from Cushman-Roisin and Beckers eq (13.7)
#   (signs implemented later)
PolRel = {'u': A*(g*omega*kz)/(N0**2*kx),
          'w': A*(g*omega)/(N0**2),
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
problem.parameters['T'] = T # period of oscillation
problem.parameters['nT'] = nT # number of periods for the ramp
problem.parameters['z_top'] = z_t

# Windowing
if SIM_TYPE==0:
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
    problem.substitutions['window'] = "1" # effectively, no window

# Ramp in time
problem.substitutions['ramp'] = "(1/2)*(tanh(4*t/(nT*T) - 2) + 1)"

# Substitutions for boundary forcing (see C-R & B eq 13.7)
problem.substitutions['fu'] = "-BFu*sin(kx*x + kz*z - omega*t)*window*ramp"
problem.substitutions['fw'] = " BFw*sin(kx*x + kz*z - omega*t)*window*ramp"
problem.substitutions['fb'] = "-BFb*cos(kx*x + kz*z - omega*t)*window*ramp"
#problem.substitutions['fp'] = "-BFp*sin(kx*x + kz*z - omega*t)*window*ramp"

###############################################################################
# Plotting function for sponge layer, background profile, etc.
#   Only plots the part handled by rank==0
def test_plot(vert, hori, plt_title, x_label, y_label, y_lims):
    matplotlib.use('Agg')
    scale = 2.5
    image = plot_tools.Box(1, 1) # aspect ratio of figure
    pad = plot_tools.Frame(0.2, 0.2, 0.15, 0.15)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)
    # Create multifigure
    mfig = plot_tools.MultiFigure(1, 1, image, pad, margin, scale)
    fig = mfig.figure
    ax = mfig.add_axes(0, 0, [0, 0, 1, 1])
    #fg, ax = plt.subplots(1,1)
    ax.set_title(plt_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lims)
    ax.plot(hori, vert, 'k-')
    plt.grid(True)
    return fig
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
    SL_array = sponge_profile(z, z_sb, z_b, sp_slope, max_sp)
else:
    SL_array = z*0.0 + 1.0
SL['g'] = SL_array
problem.parameters['SL'] = SL  # pass function in as a parameter
#   Multiply nu by SL in the equations of motion
del SL

# Plots the sponge layer coefficient profile
plt.ioff()
vert = np.array(z[0])
hori = np.array(SL_array[0])
plt_title = 'Sponge Profile'
x_label = r'Horizontal viscosity ($\nu_x$)'
y_label = r'Depth ($z$)'
y_lims  = [z_sb,z_t]
fg = test_plot(vert, hori, plt_title, x_label, y_label, y_lims)
fg.savefig('sp_layer.png')
if (plot_SL and rank == 0 and LOC):
    plt.show()

###############################################################################

# Background Profile (BP) as an NCC
print_arrays = False
BP = domain.new_field()
BP.meta['x']['constant'] = True  # means the NCC is constant along x
# Construct profile in an array so it can be used later
n_layers = NI
if n_layers == 0:
    BP_array = z*0 + 1.0
else: # Construct a staircase profile
    # Import the staircase function from the background profile script
    sys.path.insert(0, './_background_profile')
    st_top = float(params.stair_top)         # Top of staircase (not domain)
    st_bot = TEST_P #float(params.stair_bot)         # Bottom of staircase (not domian)
    if SIM_TYPE==0:
        slope = float(params.profile_slope)
        N_1 = float(params.N_1)                  # Stratification value above staircase
        N_2 = float(params.N_2)                  # Stratification value below staircase
        from Foran_profile import Foran_profile
        BP_array = Foran_profile(z, n_layers-1, st_bot, st_top, slope, N_1, N_2)

    else:
        slope = params.bp_slope
        bump = params.bump
        bg_height = params.bg_height
        from background_profile import N2_profile
        BP_array = N2_profile(z, n_layers-1, bg_height, st_bot, st_top, slope, bump)
BP['g'] = BP_array
problem.parameters['BP'] = BP  # pass function in as a parameter
del BP

# Plots the background profile
vert = np.array(z[0])
hori = np.array(BP_array[0])
plt_title = 'Background Profile'
x_label = r'Frequency ($N^2$)'
y_label = r'Depth ($z$)'
y_lims  = [z_b,z_t]
fg = test_plot(vert, hori, plt_title, x_label, y_label, y_lims)
fg.savefig('bgpf.png')
if (plot_BP and rank == 0 and LOC):
    plt.show()

###############################################################################
###############################################################################

# Equations of motion (non-linear terms on RHS)
#   Mass conservation equation
problem.add_equation("dx(u) + wz = 0")
#   Equation of state (in terms of buoyancy)
problem.add_equation("dt(b) - KA*(dx(dx(b)) + dz(bz))"
                    + "= -((N0*BP)**2)*w - (u*dx(b) + w*bz)")
#   Horizontal momentum equation
problem.add_equation("dt(u) -SL*NU*dx(dx(u)) - NU*dz(uz) + dx(p)"
                    + "= - (u*dx(u) + w*uz)")
#   Vertical momentum equation
problem.add_equation("dt(w) -SL*NU*dx(dx(w)) - NU*dz(wz) + dz(p) - b"
                    + "= - (u*dx(w) + w*wz)")

# Required for solving differential equations in Chebyshev dimension
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
if (save_all_snapshots or SIM_TYPE==0):
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
# Some other linear criterion
flow.add_property("dx(u)/omega", name='Lin_Criterion')

###############################################################################
# Outputting background profile structure to file

current_bgpf_path = '_background_profile/current_bgpf'
current_bgpf = solver.evaluator.add_file_handler(current_bgpf_path, sim_dt=sim_time_stop, max_writes=100)
# Adding a task to record the background profile for the current simulation
current_bgpf.add_task("N0*BP", layout='g', name='N')

###############################################################################
# Measuring "energy flux" through a horizontal boundary at some z

# Top level path for all energy flux snapshots
ef_snapshots_path = 'ef_snapshots'

# Adding new file handlers
if (save_all_snapshots or SIM_TYPE==1):
    # Total energy flux measurement
    ef_snapshots = solver.evaluator.add_file_handler(ef_snapshots_path, sim_dt=0.25, max_writes=100)
    # Adding a task to integrate energy flux across x for values of z
    ef_snapshots.add_task("integ(0.5*(w*u**2 + w**3) + p*w - NU*(u*uz + w*wz), 'x')", layout='g', name='<ef>')

    # Auxilary energy snapshots
    #   prescribed advec ef: "p_ef_k"      "integ(0.5*(BFw**3)*((kx/kz)**2 + 1)*sin(kx*x + kz*z_top - omega*t)*ramp, 'x')"
    aux_snaps = ["ef_advec", "ef_press", "ef_visc"]
    aux_exprs = ["integ(0.5*(w*u**2 + w**3), 'x')", "integ(p*w, 'x')", "integ(-NU*(u*uz + w*wz), 'x')"]
    aux_solver = [0, 1, 2]
    for i in range(len(aux_snaps)):
        file_path = ef_snapshots_path + '/' + aux_snaps[i]
        aux_solver[i] = solver.evaluator.add_file_handler(file_path, sim_dt=0.25, max_writes=100)
        temp_name = '<' + aux_snaps[i] + '>'
        aux_solver[i].add_task(aux_exprs[i], layout='g', name=temp_name)

###############################################################################

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        # Adaptive time stepping turned on/off in params file
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
