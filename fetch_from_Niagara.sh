#!/bin/bash
# A bash script to fetch the snapshot folders from Niagara
# Requires arguments:
#	$ sh fetch_from_Niagara.sh -j <jobname>

while getopts j: option
do
	case "${option}"
		in
		j) JOBNAME=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$JOBNAME" ]
then
	echo "No filename specified. Abort script"
	exit 1
fi

if [ -e snapshots ]
then
    echo "Removing snapshots"
    rm -r snapshots
fi
if [ -e ef_snapshots ]
then
    echo "Removing energy flux snapshots"
    rm -r ef_snapshots
fi

REMOTE_DIR=/Dedalus/Dedalus_Files/$JOBNAME
LOCAL_DIR=~/Documents/Dedalus_Projects/2D_boundary_forced_waves_v2/

# Script isn't giving password prompt when running. Weird
scp mschee@niagara.scinet.utoronto.ca:$REMOTE_DIR/slurm-1546233.out
 $LOCAL_DIR
