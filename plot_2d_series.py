"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series.py <files>... [--output=<dir>] [--ND1=<A1>] [--ND2=<B2>] [--ND3=<C3>] [--ND4=<D4>]

Options:
    --output=<dir>          Output directory [default: ./frames]
    --ND1=<A1>              Dimensionless number 1
    --ND2=<B2>              Dimensionless number 2
    --ND3=<C3>              Dimensionless number 3
    --ND4=<D4>              Dimensionless number 4

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

# Placeholders for the 3 dimensionless numbers
A1 = 1.1
B2 = 2.2
C3 = 3.3
D4 = 4.4

def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    tasks = ['b', 'p', 'u', 'w']
    scale = 2.5
    dpi = 100
    title_func = lambda sim_time: 'A1={:.2E}, B2={:.2E}, C3={:.2E}, D4={:.2E}, t={:.3f}'.format(A1, B2, C3, D4, sim_time)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout
    nrows, ncols = 4, 1
    image = plot_tools.Box(4, 1)
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
                plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True)
            # Add time title
            title = title_func(file['scales/sim_time'][index])
            title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
            fig.suptitle(title, x=0.3, y=title_height, ha='left')
            # Save figure
            savename = savename_func(file['scales/write_number'][index])
            savepath = output.joinpath(savename)
            fig.savefig(str(savepath), dpi=dpi)
            fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    A1 = float(args['--ND1'])
    B2 = float(args['--ND2'])
    C3 = float(args['--ND3'])
    D4 = float(args['--ND4'])
    if (rank==0):
        print('plot A1=',A1)
        print('plot B2=',B2)
        print('plot C3=',C3)
        print('plot D4=',D4)
    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)
