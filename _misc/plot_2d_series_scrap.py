"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series_scrap.py <files>... [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./frames]

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

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Change the size of the text overall
    font = {'size' : 16}
    plt.rc('font', **font)

    # Plot settings
    tasks = ['w']
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: r't={:.3f}'.format(sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 1, 1
    image = plot_tools.Box(3, 1)
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
                plot_bot_3d_mod(dset, 0, index, y_lims=[-0.5,0.5], axes=axes, title=task, even_scale=True) # clim=(cmin,cmax) # specify constant colorbar limits
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

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
