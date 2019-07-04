#!/bin/bash
# A bash script to scp the snapshots folder from Niagara to local
#   then create a gif

if [ -e snapshots ]
then
    echo "Removing snapshots"
    rm -rf snapshots
fi

scp -r mschee@niagara.scinet.utoronto.ca:/scratch/n/ngrisoua/mschee/Dedalus_Scratch/2D_boundary_forced_waves_v2/snapshots ./snapshots

sh run.sh -v 3 -l 0 -c 2
