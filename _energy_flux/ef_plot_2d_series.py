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

# Modified the dedalus plot function to accept y limits
import sys
sys.path.insert(0, './_misc')
from plot_tools_mod import plot_bot_3d_mod

from mpi4py import MPI
comm = MPI.COMM_WORLD
#print("thread %d of %d" % (comm.Get_rank(), comm.Get_size()))
size = comm.Get_size()
rank = comm.Get_rank()

t_0 =  0.0
t_f = 25.0
z_b = -0.5
z_t =  0.5
ratio = 0.4

with h5py.File("ef_snapshots/ef_snapshots_s1.h5", mode='r') as file:
    ef = file['tasks']['<ef>']
    print(ef.shape)
    print(ef)
    # set up multifigure
    fig, (ax0, ax1) = plt.subplots(2,1, sharex=True, figsize=(8,12), constrained_layout=False)

    # Use my modified plotbot to make heat map of EF
    paxes, caxes = plot_bot_3d_mod(ef, 1, 0, axes=ax0, y_lims=[-0.5,0.5], title='Energy flux', even_scale=True)
    # Reshape the ef object to put just the top z in an array
    ef = np.rot90(ef[:, 0, :])
    top_ef = ef[0, :]
    n_t = len(top_ef)
    t = np.linspace(t_0, t_f, n_t)

    ax1.plot(t, top_ef)
    ax1.set_title('Top boundary energy flux')
    ax1.set_xlabel('t')
    ax1.set_ylabel('<ef>')
    ax1.set_xlim(t_0, t_f)
    ax1.get_shared_x_axes().join(ax0, ax1)
    '''
    #ef = np.flipud(ef[:, 0, :].T)

    print('Grid space:',ef.shape)

    #c = ax0.pcolormesh(ef, cmap='coolwarm')

    c = ax0.imshow(ef, cmap='coolwarm', extent=[t_0, t_f, z_b, z_t])
    ax0.set_title('Energy flux')
    ax0.set_xlabel('time')
    ax0.set_ylabel('depth (z)')
    ax0.set_xlim(t_0, t_f)
    ax0.xaxis.set_tick_params(labelbottom=True)
    fig.colorbar(c, ax=ax0, orientation='horizontal', fraction=0.05)
    L_z = z_t - z_b
    L_t = t_f - t_0
    ax0.set_aspect(ratio*(L_t/L_z), adjustable='box')

    top_ef = ef[0, :]
    n_t = len(top_ef)
    t = np.linspace(t_0, t_f, n_t)
    ax1.plot(t, top_ef)
    ax1.set_title('Top boundary energy flux')
    ax1.set_xlabel('time')
    ax1.set_ylabel('EF')
    tef_min = min(top_ef)
    tef_max = max(top_ef)
    tef_range = tef_max - tef_min
    ax1.set_aspect(ratio*(L_t/tef_range), adjustable='box')
    '''
    plt.show()
# add plot of top boundary ef in subplot side by side

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
