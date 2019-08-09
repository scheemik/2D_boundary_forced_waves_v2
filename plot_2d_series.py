"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series.py LOC SIM_TYPE NI TEST_P BGPF <files>... [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./frames]
    LOC                 # 1 if local, 0 if on Niagara
    SIM_TYPE            # -e, Simulation type: (1)Energy flux or (0) reproducing run
    NI		            # Number of inner interfaces
    TEST_P	            # Test parameter
    BGPF                # path to background profile snapshots

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
#print("thread %d of %d" % (comm.Get_rank(), comm.Get_size()))
size = comm.Get_size()
rank = comm.Get_rank()

###############################################################################
# Switchboard
plot_all = False
print_args = False
contours = False
print_arrays = False
normalize_values = True
###############################################################################

# Strings for the parameters
str_ar = 'Aspect ratio'
str_nl = r'$n_{layers}$'
str_test = r'Test param'
str_om = r'$\omega$'
str_am = r'$A$'

###############################################################################

def set_title_save(fig, output, file, index, dpi, title_func, savename_func):
    # Add time title
    title = title_func(file['scales/sim_time'][index])
    fig.suptitle(title, fontsize='large')
    # Save figure
    savename = savename_func(file['scales/write_number'][index])
    savepath = output.joinpath(savename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()
    return savepath

def normalize(xmesh, ymesh, data, paramsfile):
    if paramsfile is not None:
        A = paramsfile.forcing_amp
        omega = paramsfile.omega
        g = 9.81 # [m/s^2]   Acceleration due to gravity
        N = paramsfile.N_0
        return_data = data * (N**2) / (g*omega*A)
    else:
        return_data = data
    return xmesh, ymesh, return_data

def main(filename, start, count, output):
    # Data for plotting background stratification profile
    if plot_all==False:
        bgpf_filepath = bgpf_dir + "/current_bgpf_s1.h5"
        with h5py.File(bgpf_filepath, mode='r') as file:
            bgpf = file['tasks']['N']
            temp = bgpf[()]
            hori = temp[0][0]
            z_ = file['scales']['z']['1.0']
            vert = z_[()]
        if (rank==0 and print_arrays):
            print('hori')
            print(hori)
            print('vert')
            print(vert)

    """Save plot of specified tasks for given range of analysis writes."""
    # Change the size of the text overall
    font = {'size' : 12}
    plt.rc('font', **font)

    # Fetched parameters from correct params file
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
    omega = params.omega
    A = params.forcing_amp
    T = params.T
    z_b, z_t = params.z_b, params.z_t

    # Format the dimensionless numbers nicely
    n_l   = NI
    Om    = latex_exp(omega)
    Am    = latex_exp(A)
    testp = latex_exp(TEST_P)

    # Plot settings
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: r'{:}, {:}={:}, {:}={:}, {:}={:}, {:}={:}, $t/T$={:2.3f}'.format(str_loc, str_test, testp, str_nl, n_l, str_om, Om, str_am, Am, sim_time/T)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    #   nrows, ncols set above
    pad = plot_tools.Frame(0.2, 0.2, 0.15, 0.15)
    margin = plot_tools.Frame(0.3, 0.2, 0.1, 0.1)

    # Plot settings
    # Find aspect ratio from params file
    AR = float(params.aspect_ratio)
    image = plot_tools.Box(AR, 1)
    if plot_all:
        # Specify tasks to plot
        tasks = ['b', 'p', 'u', 'w']
        nrows, ncols = 2, 2
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
                    plot_bot_3d_mod(dset, 0, index, y_lims=[z_b, z_t], axes=axes, title=task, even_scale=True, plot_contours=contours) # clim=(cmin,cmax) # specify constant colorbar limits
                set_title_save(fig, output, file, index, dpi, title_func, savename_func)
        plt.close(fig)
    else: # Just plotting background profile and w
        # Display limits for plot
        x_limits = params.x_limits
        # Plot data and parameters for background profile
        dis_ratio = 6.0 # Profile plot gets skinnier as this goes up
        xleft  = min(hori)
        xright = max(hori)
        ybott  = min(vert)
        ytop   = max(vert)
        if (xright-xleft == 0):
            xleft  =  0.0
            xright =  1.5
        calc_ratio = abs((xright-xleft)/(ybott-ytop))*dis_ratio
        task = 'w'
        nrows, ncols = 1, 2
        # Create multifigure
        mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
        fig = mfig.figure
        # Plot writes
        with h5py.File(filename, mode='r') as file:
            for index in range(start, start+count):
                # Plot vertical velocity animation on right
                axes1 = mfig.add_axes(0, 1, [0, 0, 1, 1])
                # Call 3D plotting helper, slicing in time
                dset = file['tasks'][task]
                if normalize_values:
                    plot_bot_3d_mod(dset, 0, index, x_lims=x_limits, y_lims=[z_b, z_t], axes=axes1, title=r'$wN^2/(Ag\omega)$', even_scale=True, func=normalize, paramfile=params, plot_contours=contours) # clim=(cmin,cmax) # specify constant colorbar limits
                else:
                    plot_bot_3d_mod(dset, 0, index, x_lims=x_limits, y_lims=[z_b, z_t], axes=axes1, title=r'$w$ (m/s)', even_scale=True, plot_contours=contours) # clim=(cmin,cmax) # specify constant colorbar limits

                # Plot stratification profile on the left
                axes0 = mfig.add_axes(0, 0, [0, 0, 1.5, 1])#, sharey=axes1)
                axes0.set_title('Profile')
                axes0.set_xlabel(r'$N$ (s$^{-1}$)')
                axes0.set_ylabel(r'$z$ (m)')
                axes0.set_ylim([z_b,z_t+0.04]) # fudge factor to line up y axes
                axes0.set_xlim([xleft,xright+0.04])
                axes0.plot(hori, vert, 'k-')
                # Force display aspect ratio
                axes0.set_aspect(calc_ratio)
                # save image
                imagefile = set_title_save(fig, output, file, index, dpi, title_func, savename_func)
                # crop image
        plt.close(fig)

# Not sure why, but this block needs to be at the end of the script
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
    bgpf_dir = str(args['BGPF'])
    if (rank==0 and print_args):
        print('plot',str_loc)
        print('plot',str_nl,'=',NI)
        print('plot',str_test,'=',TEST_P)
    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
