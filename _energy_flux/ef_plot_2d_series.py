"""
Plot planes from joint energy flux analysis files.

Usage:
    ef_plot_2d_series.py LOC SIM_TYPE NU KA NL <files>... [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./_energy_flux]
    LOC                 # 1 if local, 0 if on Niagara
    SIM_TYPE            # -e, Simulation type: (1)Energy flux or (0) reproducing run
    NU		            # [m^2/s]   Viscosity (momentum diffusivity)
    KA		            # [m^2/s]   Thermal diffusivity
    NL		            # [nondim]	Number of inner interfaces

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
str_nu = r'$\nu$'
str_ka = r'$\kappa$'
str_nl = r'$n_{layers}$'

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
    NU = float(args['NU'])
    KA = float(args['KA'])
    NL = int(args['NL'])
    print_vals = False
    if (rank==0 and print_vals):
        print('plot',str_loc)
        print('plot',str_nu,'=',NU)
        print('plot',str_ka,'=',KA)
        print('plot',str_nl,'=',NL)
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

t_0 =  0.0
t_f = params.sim_time_stop
#print('end time=', t_f)
z_b = -0.5
z_t =  0.0

###############################################################################

with h5py.File("ef_snapshots/ef_snapshots_s1.h5", mode='r') as file:
    # Format the dimensionless numbers nicely
    Nu    = latex_exp(NU)
    Ka    = latex_exp(KA)
    n_l   = NL

    ef = file['tasks']['<ef>']
    #print(ef.shape)
    #print(ef)
    # set up multifigure
    fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(8,12), constrained_layout=False)

    # Use my modified plotbot to make heat map of EF
    paxes, caxes = plot_bot_3d_mod(ef, 1, 0, axes=ax0, y_lims=[z_b,z_t], title='Energy flux', even_scale=True)
    # Reshape the ef object to put just the top z in an array
    ef = np.rot90(ef[:, 0, :])
    top_ef = ef[0, :]
    n_t = len(top_ef)
    t = np.linspace(t_0, t_f, n_t)
    omega = np.cos(np.pi/4)
    t_p = t*omega/(2*np.pi)
    t_0p = t_0*omega/(2*np.pi)
    t_fp = t_f*omega/(2*np.pi)

    ax1.plot(t_p, top_ef)
    ax1.set_title('Top boundary energy flux', fontsize='medium')
    ax1.set_xlabel(r'Oscilation periods $(t/T)$')
    y_label = r'Vertical energy flux $<F_z(z)>$'
    ax1.set_ylabel(y_label)
    ax1.set_xlim(t_0p, t_fp)
    ax1.get_shared_x_axes().join(ax0, ax1)

    title = r'{:}, {:}={:}, {:}={:}, {:}={:}'.format(str_loc, str_nu, Nu, str_ka, Ka, str_nl, n_l)
    fig.suptitle(title, fontsize='large')
    fig.savefig('./_energy_flux/ef_test.png', dpi=100)
# add plot of top boundary ef in subplot side by side
