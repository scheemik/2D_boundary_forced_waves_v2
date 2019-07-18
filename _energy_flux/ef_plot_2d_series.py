"""
Plot planes from joint energy flux analysis files.

Usage:
    ef_plot_2d_series.py LOC AR NU KA R0 N0 NL <files>... [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./_energy_flux]
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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from dedalus.extras import plot_tools

from mpi4py import MPI
comm = MPI.COMM_WORLD
#print("thread %d of %d" % (comm.Get_rank(), comm.Get_size()))
size = comm.Get_size()
rank = comm.Get_rank()

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
with h5py.File("ef_snapshots/ef_snapshots_s1.h5", mode='r') as file:
    ef = file['tasks']['<ef>']
    ef = ef[:, 0, :].T
    print(ef.shape)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(ef, cmap='coolwarm')
    ax.set_title('Energy flux')
    ax.set_xlabel('time')
    ax.set_ylabel('depth (z)')
    fig.colorbar(c, ax=ax)
    plt.show()


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
    AR = float(args['AR'])
    NU = float(args['NU'])
    KA = float(args['KA'])
    R0 = float(args['R0'])
    N0 = float(args['N0'])
    NL = int(args['NL'])
    print_vals = False
    if (rank==0 and print_vals):
        print('plot',str_loc)
        print('plot',str_ar,'=',AR)
        print('plot',str_nu,'=',NU)
        print('plot',str_ka,'=',KA)
        print('plot',str_r0,'=',R0)
        print('plot',str_n0,'=',N0)
        print('plot',str_nl,'=',NL)
    output_path = pathlib.Path(args['--output']).absolute()
