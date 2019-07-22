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

DATE=`date +"%m/%d-%H:%M"`
# create a 2 digit version of CORES
printf -v CO "%02d" $CORES
JOBNAME="n$CO-$DATE-2D_BF"
DIRECTORY='2D_boundary_forced_waves_v2'

set -x # echos each command as it is executed

# Go into directory of job to run
cd ${HOME}/Dedalus/Dedalus_Files/$DIRECTORY
# Pull from github the latest version of that project
git pull
# Create new directory for job
mkdir ${SCRATCH}/Dedalus/Dedalus_Files/$JOBNAME
# Copy that into the scratch directory, ignoring the .git/ directory and others
rsync -av --progress /home/n/ngrisoua/mschee/Dedalus/Dedalus_Files/2D_boundary_forced_waves_v2 /scratch/n/ngrisoua/mschee/Dedalus/Dedalus_Files/$JOBNAME --exclude .git/ --exclude mp4s/ --exclude _boundary_forcing/ --exclude _code_checkpnts/ --exclude _energy_flux/ --exclude _misc/ --exclude snapshots/ --exclude ef_snapshots/ --exclude frames/ --exclude gifs/*
cd ${SCRATCH}/Dedalus/Dedalus_Files/$JOBNAME

# Submit the job
sbatch --job-name=$JOBNAME lanceur.slrm -c $CORES

# Check the queue
squeue -u mschee

echo 'Done'
