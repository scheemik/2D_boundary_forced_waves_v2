#!/bin/bash
# A bash script to run the Dedalus python code
# Optionally takes in arguments:
#	$ sh run.sh -n <name of particular run>
#				-c <cores>
#				-e <EF(1) or reproducing run(0)>
#				-i <number of interfaces in background profile>
#				-l <local(1) or Niagara(0)>
#				-v <version: what scripts to run>
#				-k <keep(1) or allow overwriting(0)>

# if:
# VER = 0 (Full)
#	-> run the script, merge, plot EF if relevant, plot frames, create gif, create mp4
# VER = 1
#	-> run the script, merge, and plot EF if relevant
# VER = 2
#	-> run the script
# VER = 3
# 	-> merge, plot EF if relevant. plot frames, and create a gif
# VER = 4
#	-> create mp4 from frames

# Define physical parameters
NU=1.0E-6		# [m^2/s]   Viscosity (momentum diffusivity)
KA=1.4E-7		# [m^2/s]   Thermal diffusivity

while getopts n:c:e:i:l:v:k: option
do
	case "${option}"
		in
		n) NAME=${OPTARG};;
		c) CORES=${OPTARG};;
		e) SIM_TYPE=${OPTARG};;
		i) NI=${OPTARG};;
		l) LOC=${OPTARG};;
		v) VER=${OPTARG};;
		k) KEEP=${OPTARG};;
	esac
done

# check to see if arguments were passed
if [ -z "$NAME" ]
then
	NAME=`date +"%m-%d_%Hh%M"`
	echo "-n, No name specified, using $NAME"
fi
if [ -z "$CORES" ]
then
	CORES=2
	echo "-c, No number of cores specified, using CORES=$CORES"
fi
if [ -z "$SIM_TYPE" ]
then
	echo "-e, No simulation type specified, running energy flux measurements"
	SIM_TYPE=1
fi
if [ -z "$NI" ]
then
	NI=1
	echo "-i, No number of interfaces specified, using NI=$NI"
fi
if [ -z "$LOC" ]
then
	LOC=1 # 1 if local, 0 if on Niagara
	echo "-l, No version specified, using LOC=$LOC"
fi
if [ -z "$VER" ]
then
	VER=1
	echo "-v, No version specified, using VER=$VER"
fi
if [ -z "$KEEP" ]
then
	KEEP=0
	echo "-k, No 'keep' preference specified, using KEEP=$KEEP"
fi

###############################################################################
# keep - create archive directory for experiment
#	if (KEEP = 1 and VER != 4)

if [ $KEEP -eq 1 ] && [ $VER -ne 3 ] && [ $VER -ne 4 ]
then
	echo ''
	echo '--Preparing archive directory--'
	echo ''
	# Check if experiments folder exists
	if [ -e _experiments ]
	then
		continue
	else
		mkdir ./_experiments
	fi
	# Check if this experiment was run before
	if [ -e _experiments/$NAME ]
	then
		echo 'Experiment run previously. Overwriting data'
		rm -rf _experiments/$NAME
	else
		continue
	fi
	# Make a new directory under the experiments folder
	mkdir ./_experiments/$NAME
	# Create experiment log file
	LOG_FILE=_experiments/$NAME/LOG_${NAME}.txt
	touch $LOG_FILE
	echo "--Run options:" >> $LOG_FILE
	echo "" >> $LOG_FILE
	echo "-n, Experiment name = ${NAME}" >> $LOG_FILE
	echo "-c, Number of cores = ${CORES}" >> $LOG_FILE
	if [ $SIM_TYPE -eq 1 ]
	then
		echo "-e, (${SIM_TYPE}) Simulation for measuring energy flux" >> $LOG_FILE
	else
		echo "-e, (${SIM_TYPE}) Simulation for reproducing results" >> $LOG_FILE
	fi
	echo "-i, Number of interfaces = ${NI}" >> $LOG_FILE
	if [ $LOC -eq 1 ]
	then
		echo "-l, (${LOC}) Simulation run on local pc" >> $LOG_FILE
	else
		echo "-l, (${LOC}) Simulation run on Niagara" >> $LOG_FILE
	fi
	echo "-v, Version of run = ${VER}" >> $LOG_FILE
	echo "" >> $LOG_FILE

	# Write out from parameter file
	python3 write_out_params.py $LOC $NAME $SIM_TYPE
fi

###############################################################################
# run the script
#	if (VER = 0, 1, 2)

if [ $VER -eq 0 ] || [ $VER -eq 1 ] || [ $VER -eq 2 ]
then
	echo ''
	echo '--Running script--'
	echo ''
	# 	Check to see if the frames and snapshots folders exist
	#	If so, remove them before running the dedalus script
	if [ -e frames ]
	then
		echo "Removing frames"
		rm -r frames
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
	if [ -e _background_profile/current_bgpf ]
	then
		echo "Removing background profile snapshot"
		rm -r _background_profile/current_bgpf
	fi

	# If running on local pc
	if [ $LOC -eq 1 ]
	then
		echo "Running Dedalus script for local pc"
		# mpiexec uses -n flag for number of processes to use
	    mpiexec -n $CORES python3 current_code.py $LOC $SIM_TYPE $NU $KA $NI
	    echo ""
	fi
	# If running on Niagara
	if [ $LOC -eq 0 ]
	then
		echo "Running Dedalus script for Niagara"
		# mpiexec uses -n flag for number of processes to use
	    mpiexec -n $CORES python3.6 current_code.py $LOC $SIM_TYPE $NU $KA $NI
		echo ""
	fi
fi

###############################################################################
# Setting paths to outputs

# If VER = 3, change paths to snapshot directories
if [ $VER -eq 3 ]
then
	snapshot_path=_experiments/$NAME/snapshots
	background_snapshot_path=_experiments/$NAME/current_bgpf
	ef_snapshot_path=_experiments/$NAME/ef_snapshots
	frames_path=_experiments/$NAME/frames
else
	snapshot_path=snapshots
	background_snapshot_path=_background_profile/current_bgpf
	ef_snapshot_path=ef_snapshots
	frames_path=frames
fi

###############################################################################
# merge and plot EF if relevant
#	if (VER = 0, 1, 3)

if [ $VER -eq 0 ] || [ $VER -eq 1 ] || [ $VER -eq 3 ]
then
	echo ''
	echo '--Merging--'
	echo ''
	# Check to make sure snapshots folder exists
	if [ -e $snapshot_path ]
	then
		continue
	else
		echo "Cannot find snapshots. Aborting script"
		exit 1
	fi
	# Check if snapshots have already been merged
	if [ -e $snapshot_path/snapshots_s1.h5 ]
	then
		echo "Snapshots already merged"
	else
		echo "Merging snapshots"
		mpiexec -n $CORES python3 merge.py $snapshot_path
	fi
	# Check if background profile snapshots have already been merged
	if [ -e $background_snapshot_path/current_bgpf_s1.h5 ]
	then
		echo "Background profile snapshot already merged"
	else
		echo "Merging background profile snapshot"
		mpiexec -n $CORES python3 merge.py $background_snapshot_path
	fi
	# Check if energy flux snapshots have already been merged
	if [ -e $ef_snapshot_path/ef_snapshots_s1.h5 ]
	then
		echo "Energy flux snapshots already merged"
	else
		echo "Merging energy flux snapshots"
		mpiexec -n $CORES python3 merge.py $ef_snapshot_path
	fi

	# Check if plotting energy flux if relevant
	if [ $SIM_TYPE -eq 1 ] #|| [ 1 -eq 1 ]
	then
		echo ''
		echo "Plotting EF for z vs. t"
		mpiexec -n $CORES python3 _energy_flux/ef_plot_2d_series.py $LOC $SIM_TYPE $NU $KA $NI $ef_snapshot_path/*.h5
	fi
fi

###############################################################################
# plot frames
#	if (VER = 0, 3)

if [ $VER -eq 0 ] || [ $VER -eq 3 ]
then
	echo ''
	echo '--Plotting frames--'
	echo ''
	if [ -e frames ]
	then
		echo "Removing old frames"
		rm -rf frames
	fi
	echo "Plotting 2d series"
	mpiexec -n $CORES python3 plot_2d_series.py $LOC $SIM_TYPE $NU $KA $NI $background_snapshot_path $snapshot_path/*.h5
fi

###############################################################################
# create gif
#	if (VER = 0, 3)

if [ $VER -eq 0 ] || [ $VER -eq 3 ]
then
	echo ''
	echo '--Creating gif--'
	echo ''
	# Delete old test gif if it exists
	if [ -e gifs/test.gif ]
	then
		echo "Deleting test.gif"
		rm -rf gifs/test.gif
		echo ""
	fi
	echo 'Checking frames'
	files=/$frames_path/*
	if [ -e $frames_path ] && [ ${#files[@]} -gt 0 ]
	then
		echo "Executing gif script"
		python3 create_gif.py gifs/test.gif $frames_path
	else
		echo "No frames found"
	fi
fi

###############################################################################
# create mp4
#	if (VER = 0, 4)

if [ $VER -eq 0 ] || [ $VER -eq 4 ]
then
	echo ''
	echo '--Creating mp4--'
	echo ''
	# Check if frames exist
	echo 'Checking frames'
	files=/$frames_path/*
	if [ -e $frames_path ] && [ ${#files[@]} -gt 0 ]
	then
		echo "Executing mp4 command"
		cd $frames_path/
		ffmpeg -framerate 10 -i write_%06d.png -c:v libx264 -pix_fmt yuv420p test.mp4
		cd ..
		mv $frames_path/test.mp4 ./
	else
		echo "No frames found"
	fi
fi

###############################################################################
# keep - rename outputs to archive simulation in separate folder
#	if (KEEP = 1 and VER != 4)

if [ $KEEP -eq 1 ] && [ $VER -ne 4 ]
then
	echo ''
	echo '--Archiving outputs--'
	echo ''
	# Check if experiment folder exists
	if [ -e _experiments/$NAME ]
	then
		continue
	else
		echo "Archive directory not found. Aborting script"
		exit 1
	fi
	# New snapshots if (VER = 0, 1, 2, 3)
	if [ $VER -eq 0 ] || [ $VER -eq 1 ] || [ $VER -eq 2 ] || [ $VER -eq 3 ]
	then
		# Check if snapshots exist
		if [ -e $snapshot_path ]
		then
			# Move snapshots to new directory
			mv $snapshot_path/ _experiments/$NAME/
			# Copy `merge.py` script to new directory
			cp merge.py _experiments/$NAME/
		else
			echo "No snapshots found to archive. Aborting script"
			exit 0
		fi
		# Check if background profile snapshots exist
		if [ -e $background_snapshot_path ]
		then
			# Move snapshots to new directory
			mv $background_snapshot_path _experiments/$NAME/
		fi
		# Check if energy flux snapshots exist
		if [ -e $ef_snapshot_path ]
		then
			# Move snapshots to new directory
			mv $ef_snapshot_path/ _experiments/$NAME/
		fi
	fi
	# Plot of EF if (VER = 0, 1, 3)
	if [ $VER -eq 0 ] || [ $VER -eq 1 ] || [ $VER -eq 3 ]
	then
		# Check if energy flux plot exists
		if [ -e _energy_flux/ef_test.png ]
		then
			# Move energy flux plot to new directory
			mv _energy_flux/ef_test.png _experiments/$NAME/ef_plot.png
		fi
	else
		# Copy the energy flux plotting script to new directory
		mkdir _experiments/$NAME/_energy_flux
		cp _energy_flux/ef_plot_2d_series.py _experiments/$NAME/_energy_flux/
	fi
	# Plotted frames and made gif if (VER = 0, 3)
	if [ $VER -eq 0 ] || [ $VER -eq 3 ]
	then
		# Check if frames exist
		files=/$frames_path/*
		if [ -e $frames_path ] && [ ${#files[@]} -gt 0 ]
		then
			# Move frames to new directory
			mv $frames_path/ _experiments/$NAME/
		fi
		# Check if gif exists
		if [ -e gifs/test.gif ]
		then
			# Move gif to new directory
			mv gifs/test.gif _experiments/$NAME/${NAME}.gif
		fi
	else
		# Copy the plotting and creating gif scripts to new directory
		cp plot_2d_series.py _experiments/$NAME/
		cp create_gif.py _experiments/$NAME/
	fi
	# Created mp4 if (VER = 0)
	if [ $VER -eq 0 ]
	then
		# Check if mp4 exists
		if [ -e test.mp4 ]
		then
			# Move mp4 to new directory
			mv test.mp4 _experiments/$NAME/${NAME}.mp4
		fi
	else
		# Copy this run script to new directory
		cp run.sh _experiments/$NAME/
	fi
fi

echo "Done"
