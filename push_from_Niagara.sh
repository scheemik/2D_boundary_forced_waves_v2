#!/bin/bash
# A bash script to push the snapshot folders from Niagara
# Warning: this will overwrite the snapshot folders in
#	the local directory without checking

LOCAL_DIR=~/Documents/Dedalus_Projects/2D_boundary_forced_waves_v2/

scp -r snapshots/ mschee@ngws19.atmosp.physics.utoronto.ca:~/Documents/Dedalus_Projects/2D_boundary_forced_waves_v2/
scp -r ef_snapshots/ mschee@ngws19.atmosp.physics.utoronto.ca:~/Documents/Dedalus_Projects/2D_boundary_forced_waves_v2/
