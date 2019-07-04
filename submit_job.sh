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
JOBNAME="n$CORES-$DATE-2D_BF"
DIRECTORY='2D_boundary_forced_waves_v2'
SUBDIR=''
# create a 2 digit version of CORES to help with file naming
printf -v CO "%02d" $CORES
NSUBDIR=''#"n$CO-$SUBDIR"

set -x # echos each command as it is executed

cd ${HOME}/Dedalus_Projects/$DIRECTORY

if [ -e ${SCRATCH}/Dedalus_Scratch/$DIRECTORY/$NSUBDIR ]
then
        rm -rf ${SCRATCH}/Dedalus_Scratch/$DIRECTORY/$NSUBDIR
fi

# Go into directory of job to run
cd ${HOME}/Dedalus_Projects/$DIRECTORY
# Pull from github the latest version of that project
git pull
# Copy that into the scratch directory, ignoring the .git/ directory and others
rsync -av --progress /home/n/ngrisoua/mschee/Dedalus_Projects/2D_boundary_forced_waves_v2 /scratch/n/ngrisoua/mschee/Dedalus_Scratch --exclude .git/ --exclude mp4s/ --exclude non-dimensional/ --exclude OG_RB_code/ --exclude _boundary_forcing/ --exclude _code_checkpnts/ --exclude sympy_test.py
mv ${SCRATCH}/Dedalus_Scratch/$DIRECTORY/$SUBDIR ${SCRATCH}/Dedalus_Scratch/$DIRECTORY/$NSUBDIR
cd ${SCRATCH}/Dedalus_Scratch/$DIRECTORY/$NSUBDIR

# Submit the job
sbatch --job-name=$JOBNAME lanceur.slrm -c $CORES

# Check the queue
squeue -u mschee

echo 'Done'
