"""
Plot planes from joint energy flux analysis files.

Usage:
    ef_plot_2d_series.py LOC SIM_TYPE NI TEST_P SNAPSHOT_PATH [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./_energy_flux]
    LOC                 # 1 if local, 0 if on Niagara
    SIM_TYPE            # -e, Simulation type: (1)Energy flux or (0) reproducing run
    NI		            # [nondim]	Number of interfaces
    TEST_P	            # Test parameter
    SNAPSHOT_PATH       # Path to where the ef_snapshots are held

"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools

# Modified the dedalus plot function to accept y limits
import sys
sys.path.insert(0, './_misc')
from plot_tools_mod import plot_bot_3d_mod
from latex_fmt import latex_exp

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

###############################################################################

# Make a twin axis for top and bottom energy fluxes
#   Twin=true means they will each have their own vertical axis
twin_top_bot = True

# Plot auxiliary energy flux snapshots (3 terms)
plot_aux_ef = True

# Strings for the parameters
str_test = r'Test param'
str_nl = r'$n_{layers}$'
str_om = r'$\omega$'
str_am = r'$A$'

###############################################################################

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    LOC = int(args['LOC'])
    str_loc = 'Local' if bool(LOC) else 'Niagara'
    SIM_TYPE = int(args['SIM_TYPE'])
    NI = int(args['NI'])
    TEST_P = float(args['TEST_P'])
    ef_snapshot_path = str(args['SNAPSHOT_PATH'])
    print_vals = False
    if (rank==0 and print_vals):
        print('plot',str_loc)
        print('plot',str_test,'=',TEST_P)
        print('plot',str_nl,'=',NI)
        print('plot',str_om,'=',Om)
        print('plot',str_am,'=',Am)
    output_path = pathlib.Path(args['--output']).absolute()

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
    import params_ef
    params = params_ef

z_b, z_t =  params.z_b, params.z_t
omega = params.omega
A = params.forcing_amp
T = params.T

if LOC==1:
    import lparams_local
    lparams = lparams_local
else:
    import lparams_Niagara
    lparams = lparams_Niagara

nz = lparams.n_z
t_0  = 0.0
t_fp = lparams.sim_period_stop   #[t/T]
# Calculate stop time
t_f = t_fp * T # [seconds]

###############################################################################

merged_snapshots = ef_snapshot_path + "/ef_snapshots.h5"
# set up multifigure
fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(8,12), constrained_layout=False)
with h5py.File(merged_snapshots, mode='r') as file:
    # Format the dimensionless numbers nicely
    testp = latex_exp(TEST_P)
    n_l   = NI
    Om    = latex_exp(omega)
    Am    = latex_exp(A)

    ef = file['tasks']['<ef>']
    st = file['scales']['sim_time']
    t  = st[()]

    # Use my modified plotbot to make heat map of EF
    paxes, caxes = plot_bot_3d_mod(ef, 1, 0, axes=ax0, x_lims=[t_0,t_f], y_lims=[z_b,z_t], title='Vertical energy flux', even_scale=True)
    # Reshape the ef object to put just the top z in an array
    ef = np.rot90(ef[:, 0, :])
    top_ef = ef[0, :]
    # then the bottom z
    bot_ef = ef[nz, :]
    t_p = t*omega/(2*np.pi)

    ax1.set_title('Energy flux through boundaries', fontsize='medium')
    ax1.set_xlabel(r'Oscilation periods $(t/T)$')
    y_label = r'Vertical energy flux $<F_z(z)>$'
    ax1.set_ylabel(y_label)
    ax1.set_xlim(0.0, t_fp)

    color = 'tab:blue'
    ln1 = ax1.plot(t_p[1:], top_ef[1:], label='Top - Total measured', color=color) # initial value doesn't make sense, so skip it
    if twin_top_bot:
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ln2 = ax2.plot(t_p[1:], bot_ef[1:], label='Bottom - Total measured', color=color) # initial value doesn't make sense, so skip it
        ax2.tick_params(axis='y', labelcolor=color)
    else:
        color = 'tab:orange'
        ln2 = ax1.plot(t_p[1:], bot_ef[1:], label='Bottom - Total measured', color=color) # initial value doesn't make sense, so skip it
'''
merged_snapshots = ef_snapshot_path + "/p_ef_k/p_ef_k.h5"
with h5py.File(merged_snapshots, mode='r') as file:
    p_ef_k = file['tasks']['<p_ef_k>']
    st = file['scales']['sim_time']
    t  = st[()]
    t_p = t*omega/(2*np.pi)
    # Reshape the ef object to put just the top z in an array
    p_ef_k = np.rot90(p_ef_k[:, 0, :])
    top_p_ef_k = p_ef_k[0, :]
    #ax1.plot(t_p[1:], top_p_ef_k[1:], label='Top - Prescribed kinetic') # initial value doesn't make sense, so skip it
'''
# put together the legend
lns = ln1+ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels)
ax1.get_shared_x_axes().join(ax0, ax1)
title = r'{:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}'.format(str_loc, str_test, testp, str_nl, n_l, str_om, Om, str_am, Am)
fig.suptitle(title, fontsize='large')
fig.savefig('./_energy_flux/ef_test.png', dpi=100)


###############################################################################
# auxiliary
if plot_aux_ef == True:
    fig, (ax0, ax1, ax2) = plt.subplots(3,1, sharex=True, figsize=(8,12), constrained_layout=False)
    axes = [ax0, ax1, ax2]
    line_colors = ['tab:blue', 'tab:orange', 'tab:green']
    aux_snaps = ["ef_advec", "ef_press", "ef_visc"]
    plot_titles = ["Advection term", "Pressure term", "Viscous term"]
    for i in range(len(axes)):
        merged_snapshots = ef_snapshot_path + "/" + aux_snaps[i] + "/" + aux_snaps[i] + ".h5"
        with h5py.File(merged_snapshots, mode='r') as file:
            task_name = '<' + aux_snaps[i] + '>'
            aux_snap = file['tasks'][task_name]
            st = file['scales']['sim_time']
            t  = st[()]
            t_p = t*omega/(2*np.pi)
            # Reshape the ef object to put just the top z in an array
            aux_snap = np.rot90(aux_snap[:, 0, :])
            top_aux_snap = aux_snap[0, :]
            axes[i].plot(t_p[1:], top_aux_snap[1:], label=task_name, color=line_colors[i]) # initial value doesn't make sense, so skip it
            axes[i].set_title(plot_titles[i], fontsize='medium')
            y_label = r'Vertical energy flux $<F_z(z)>$'
            axes[i].set_ylabel(y_label)
            axes[i].set_xlim(0.0, t_fp)
            axes[i].grid(True)
    axes[len(axes)-1].set_xlabel(r'Oscilation periods $(t/T)$')
    title = r'{:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}'.format(str_loc, str_test, testp, str_nl, n_l, str_om, Om, str_am, Am)
    fig.suptitle(title, fontsize='large')
    fig.savefig('./_energy_flux/ef_aux.png', dpi=100)
