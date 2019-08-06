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

# Expects numbers in the format 7.0E+2
def latex_exp(num):
    float_str = "{:.1E}".format(num)
    if "E" in float_str:
        base, exponent = float_str.split("E")
        exp = int(exponent)
        str1 = '$' + str(base)
        if (exp != 0):
            str1 = str1 + r'\cdot10^{' + str(exp)
        str1 = str1 + '}$'
        return r"{0}".format(str1)
    else:
        return float_str

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
    import params_repro
    params = params_repro
else:
    # Measuring energy flux
    import params_ef
    params = params_ef

t_0  = 0.0
t_f  = params.sim_time_stop     #[seconds]
t_fp = params.sim_period_stop   #[t/T]
z_b, z_t =  params.z_b, params.z_t
omega = params.omega
A = TEST_P #params.forcing_amp

###############################################################################

merged_snapshots = ef_snapshot_path + "/ef_snapshots.h5"
with h5py.File(merged_snapshots, mode='r') as file:
    # Format the dimensionless numbers nicely
    testp = latex_exp(TEST_P)
    n_l   = NI
    Om    = latex_exp(omega)
    Am    = latex_exp(A)

    ef = file['tasks']['<ef>']
    st = file['scales']['sim_time']
    t  = st[()]
    # set up multifigure
    fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(8,12), constrained_layout=False)

    # Use my modified plotbot to make heat map of EF
    paxes, caxes = plot_bot_3d_mod(ef, 1, 0, axes=ax0, x_lims=[t_0,t_f], y_lims=[z_b,z_t], title='Energy flux', even_scale=True)
    # Reshape the ef object to put just the top z in an array
    ef = np.rot90(ef[:, 0, :])
    top_ef = ef[0, :]
    n_t = len(top_ef)
    #t = np.linspace(t_0, t_f, n_t)
    #omega = np.cos(np.pi/4)
    t_p = t*omega/(2*np.pi)
    #t_0p = t_0*omega/(2*np.pi)
    #t_fp = t_f*omega/(2*np.pi)

    ax1.plot(t_p[1:], top_ef[1:]) # initial value doesn't make sense, so skip it
    ax1.set_title('Top boundary energy flux', fontsize='medium')
    ax1.set_xlabel(r'Oscilation periods $(t/T)$')
    y_label = r'Vertical energy flux $<F_z(z)>$'
    ax1.set_ylabel(y_label)
    ax1.set_xlim(0.0, t_fp)
    ax1.get_shared_x_axes().join(ax0, ax1)

    title = r'{:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}'.format(str_loc, str_test, testp, str_nl, n_l, str_om, Om, str_am, Am)
    fig.suptitle(title, fontsize='large')
    fig.savefig('./_energy_flux/ef_test.png', dpi=100)
