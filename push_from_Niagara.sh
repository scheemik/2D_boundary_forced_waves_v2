#!/bin/bash
# A bash script to copy an experiment from Niagara to local
# push_from_Niagara.sh takes in these arguments:
#	$ sh push_from_Niagara.sh -n <name of particular run>

while getopts n: option
do
	case "${option}"
		in
		n) NAME=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$NAME" ]
then
	echo "No name specified, aborting script"
	exit 1
fi

EXP_PATH=_experiments/${NAME}/

LOCAL_EXP=~/Documents/Dedalus_Projects/2D_boundary_forced_waves_v2/_experiments/

scp -r $EXP_PATH mschee@ngws19.atmosp.physics.utoronto.ca:${LOCAL_EXP}
#scp -r ef_snapshots/ mschee@ngws19.atmosp.physics.utoronto.ca:~/Documents/Dedalus_Projects/2D_boundary_forced_waves_v2/
