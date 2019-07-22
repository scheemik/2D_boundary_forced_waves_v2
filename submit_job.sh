#!/bin/bash
# A bash script to submit a job to Niagara
# Optionally takes in arguments:
#	$ sh submit_profile_job.sh -c <cores>

while getopts c: option
do
	case "${option}"
		in
		c) CORES=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$CORES" ]
then
	echo "No number of cores specified, using CORES=32"
	CORES=32
fi

# Prepare scratch

DATE=`date +"%m-%d_%Hh%M"`
# create a 2 digit version of CORES
printf -v CO "%02d" $CORES
JOBNAME="$DATE-2D_BF-n$CO"
DIRECTORY='Dedalus/Vanilla_Dedalus'
SUBDIRECT='2D_boundary_forced_waves_v2'

set -x # echos each command as it is executed

# Go into directory of job to run
cd ${HOME}/$DIRECTORY/$SUBDIRECT
# Pull from github the latest version of that project
git pull
# Copy that into the scratch directory, ignoring the .git/ directory and others
cp -r ${HOME}/$DIRECTORY/$SUBDIRECT ${SCRATCH}/$DIRECTORY/$SUBDIRECT
#rsync -av --progress ${HOME}/Dedalus/Dedalus_Files/$DIRECTORY ${SCRATCH}/Dedalus/Dedalus_Files/ --exclude .git/ --exclude mp4s/ --exclude _boundary_forcing/ --exclude _code_checkpnts/ --exclude _energy_flux/ --exclude _misc/ --exclude snapshots/ --exclude ef_snapshots/ --exclude frames/ --exclude gifs/*
mv ${SCRATCH}/$DIRECTORY/$SUBDIRECT ${SCRATCH}/$DIRECTORY/$JOBNAME
cd ${SCRATCH}/$DIRECTORY/$JOBNAME
ls

# Submit the job
sbatch --job-name=$JOBNAME lanceur.slrm -c $CORES

# Check the queue
squeue -u mschee

echo 'Done'
