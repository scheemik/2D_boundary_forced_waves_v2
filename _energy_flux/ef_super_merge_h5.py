"""
Merges merged energy flux analysis files into one file.

Usage:
    ef_super_merge_h5.py SNAPSHOT_PATH [--output=<dir>]

Options:
    --output=<dir>      # Output directory [default: ./_energy_flux]
    SNAPSHOT_PATH       # Path to where the ef_snapshots are held

"""

import h5py
import numpy as np
from dedalus.extras import plot_tools
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

###############################################################################

#if __name__ == "__main__":

import pathlib
from docopt import docopt
#from dedalus.tools import logging
from dedalus.tools import post
#from dedalus.tools.parallel import Sync

args = docopt(__doc__)

ef_snapshot_path = str(args['SNAPSHOT_PATH'])

#output_path = pathlib.Path(args['--output']).absolute()

###############################################################################

# Merge all ef_snapshots into one file
merged_snapshots = ef_snapshot_path + "/ef_snapshots.h5"
if os.path.exists(merged_snapshots):
    os.remove(merged_snapshots)
set_paths = list(pathlib.Path(ef_snapshot_path).glob("ef_snapshots_s*.h5"))
post.merge_sets(merged_snapshots, set_paths, cleanup=False)

###############################################################################
