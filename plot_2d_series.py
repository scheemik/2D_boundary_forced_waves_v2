"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series.py LOC AR NU KA R0 N0 NL <files>... [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./frames]
    LOC                 # 1 if local, 0 if on Niagara
    AR		            # [nondim]  Aspect ratio of domain
    NU		            # [m^2/s]   Viscosity (momentum diffusivity)
    KA		            # [m^2/s]   Thermal diffusivity
    R0		            # [kg/m^3]  Characteristic density
    N0		            # [rad/s]   Characteristic stratification
    NL		            # [nondim]	Number of inner interfaces

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools
from mpi4py import MPI
comm = MPI.COMM_WORLD
#print("thread %d of %d" % (comm.Get_rank(), comm.Get_size()))
size = comm.Get_size()
rank = comm.Get_rank()

# Strings for the parameters
str_ar = 'Aspect ratio'
str_nu = r'$\nu$'
str_ka = r'$\kappa$'
str_r0 = r'$\rho_0$'
str_n0 = r'$N_0$'
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

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Change the size of the text overall
    font = {'size' : 16}
    plt.rc('font', **font)

    # Format the dimensionless numbers nicely
    Nu    = latex_exp(NU)
    Ka    = latex_exp(KA)
    rho_0 = latex_exp(R0)
    N_0   = latex_exp(N0)
    n_l   = NL

    # Plot settings
    tasks = ['b', 'p', 'u', 'w']
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: r'{:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}, t={:.3f}'.format(str_loc, str_nu, Nu, str_ka, Ka, str_r0, rho_0, str_n0, N_0, str_nl, n_l, sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 2, 2#4, 1
    image = plot_tools.Box(AR, 1)
    pad = plot_tools.Frame(0.2, 0.2, 0.1, 0.1)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Create multifigure
    mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
    fig = mfig.figure
    # Plot writes
    with h5py.File(filename, mode='r') as file:
        for index in range(start, start+count):
            for n, task in enumerate(tasks):
                # Build subfigure axes
                i, j = divmod(n, ncols)
                axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True) # clim=(cmin,cmax) # specify constant colorbar limits
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            fig.suptitle(title, fontsize='large')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)

# Not sure why, but this block needs to be at the end of the script
if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    LOC = bool(args['LOC'])
    str_loc = 'Local' if LOC else 'Niagara'
    AR = float(args['AR'])
    NU = float(args['NU'])
    KA = float(args['KA'])
    R0 = float(args['R0'])
    N0 = float(args['N0'])
    NL = int(args['NL'])
    if (rank==0):
        print('plot',str_loc,'=',LOC)
        print('plot',str_ar,'=',AR)
        print('plot',str_nu,'=',NU)
        print('plot',str_ka,'=',KA)
        print('plot',str_r0,'=',R0)
        print('plot',str_n0,'=',N0)
        print('plot',str_nl,'=',NL)
    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
