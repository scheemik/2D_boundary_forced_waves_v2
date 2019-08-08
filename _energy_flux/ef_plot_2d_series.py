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
    paxes, caxes = plot_bot_3d_mod(ef, 1, 0, axes=ax0, x_lims=[t_0,t_f], y_lims=[z_b,z_t], title='Energy flux', even_scale=True)
    # Reshape the ef object to put just the top z in an array
    ef = np.rot90(ef[:, 0, :])
    top_ef = ef[0, :]
    #n_t = len(top_ef)
    t_p = t*omega/(2*np.pi)

    ax1.plot(t_p[1:], top_ef[1:], label='Total measured') # initial value doesn't make sense, so skip it
    ax1.set_title('Top boundary energy flux', fontsize='medium')
    ax1.set_xlabel(r'Oscilation periods $(t/T)$')
    y_label = r'Vertical energy flux $<F_z(z)>$'
    ax1.set_ylabel(y_label)
    ax1.set_xlim(0.0, t_fp)

merged_snapshots = ef_snapshot_path + "/p_ef_k/p_ef_k.h5"
with h5py.File(merged_snapshots, mode='r') as file:
    p_ef_k = file['tasks']['<p_ef_k>']
    st = file['scales']['sim_time']
    t  = st[()]
    t_p = t*omega/(2*np.pi)
    # Reshape the ef object to put just the top z in an array
    p_ef_k = np.rot90(p_ef_k[:, 0, :])
    top_p_ef_k = p_ef_k[0, :]
    ax1.plot(t_p[1:], top_p_ef_k[1:], label='Prescribed kinetic') # initial value doesn't make sense, so skip it

ax1.legend()
ax1.get_shared_x_axes().join(ax0, ax1)
title = r'{:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}'.format(str_loc, str_test, testp, str_nl, n_l, str_om, Om, str_am, Am)
fig.suptitle(title, fontsize='large')
fig.savefig('./_energy_flux/ef_test.png', dpi=100)
