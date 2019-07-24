#!/bin/bash
# A bash script to run the Dedalus python code
# Optionally takes in arguments:
#	$ sh run.sh -v <version_number> -l <local_or_not> -c <cores>

while getopts v:l:c: option
do
	case "${option}"
		in
		v) VER=${OPTARG};;
		l) LOC=${OPTARG};;
		c) CORES=${OPTARG};;
	esac
done

echo "VER=$VER"
echo "LOC=$LOC"
echo "CORES=$CORES"

# check to see if arguments were passed
if [ -z "$VER" ]
then
	echo "No version specified, using VER=2"
	VER=2
fi
if [ -z "$LOC" ]
then
	echo "No version specified, using LOC=1"
	LOC=1 # 1 if local, 0 if on Niagara
fi
if [ -z "$CORES" ]
then
	echo "No number of cores specified, using CORES=2"
	CORES=2
fi

# if:
# VER = 1
#	-> run the script, merge, plot, and create a gif
# VER = 2
#	-> run the script
# VER = 3
# 	-> merge, plot, and create a gif
# VER = 4
#	-> create mp4 from frames
# VER = 5
#	-> merge and plot energy flux

# Define parameters
AR=1.0			# [nondim]  Aspect ratio of domain
NU=1.0E-6		# [m^2/s]   Viscosity (momentum diffusivity)
KA=1.4E-7		# [m^2/s]   Thermal diffusivity
#R0=1.0E+3		# [kg/m^3]  Characteristic density
N0=1.0E+0		# [rad/s]   Characteristic stratification
NL=1			# [nondim]	Number of interfaces

# If VER = 1 or 2, then the code will run and make snapshots
# 	Check to see if the frames and snapshots folders exist
#	If so, remove them before running the dedalus script
if [ $VER -lt 3 ]
then
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
fi

# If running on local pc and VER=1 or 2
if [ $LOC -eq 1 ] && [ $VER -lt 3 ]
then
	#echo 0 > /proc/sys/kernel/yama/ptrace_scope
	echo "Running Dedalus script for local pc"
	# mpiexec uses -n flag for number of processes to use
    mpiexec -n $CORES python3 current_code.py $LOC $AR $NU $KA $N0 $NL
    echo ""
fi

# If running on Niagara and VER=1 or 2
if [ $LOC -eq 0 ] && [ $VER -lt 3 ]
then
	echo "Running Dedalus script for Niagara"
	# mpirun uses -c, -n, --n, or -np for number of threads / cores
	#mpirun -c $CORES python3.6 current_code.py $LOC $DN1 $DN2 $DN3 $DN4
	# mpiexec uses -n flag for number of processes to use
    mpiexec -n $CORES python3.6 current_code.py $LOC $AR $NU $KA $N0 $NL
	echo ""
fi

# If VER = 1 or 3, then run the rest of the code,
# 	but first check if snapshots folder was made
#if [ $VER -ne 2 ] && [ $VER -ne 4 ] && [ $VER -ne 5 ] && [ -e snapshots ]
if ( [ $VER -eq 1 ] || [ $VER -eq 3 ] ) && [ -e snapshots ]
then
	if [ -e gifs/test.gif ]
	then
		echo "Deleting test.gif"
		rm -rf gifs/test.gif
		echo ""
	fi
	if [ -e snapshots/snapshots_s1.h5 ]
	then
		echo "Snapshots already merged"
	else
		echo "Merging snapshots"
		mpiexec -n $CORES python3 merge.py snapshots
	fi
	echo ""
	if [ -e frames ]
	then
		echo "Removing frames"
		rm -rf frames
		echo ""
	fi
	echo "Plotting 2d series"
	mpiexec -n $CORES python3 plot_2d_series.py $LOC $AR $NU $KA $N0 $NL snapshots/*.h5
	echo ""
	files=/frames/*
	if [ -e frames ] && [ ${#files[@]} -gt 0 ]
	then
		echo "Creating gif"
		python3 create_gif.py gifs/test.gif
	fi
fi

# If VER = 4, then create an mp4 video of the existing frames
if [ $VER -eq 4 ] && [ -e frames ]
then
	echo "Creating mp4"
	cd frames/
	ffmpeg -framerate 10 -i write_%06d.png -c:v libx264 -pix_fmt yuv420p test.mp4
	cd ..
	mv frames/test.mp4 mp4s/
fi

# If VER = 5, then merge and plot the energy flux
if [ $VER -eq 5 ] && [ -e ef_snapshots ]
then
	if [ -e ef_snapshots/ef_snapshots_s1.h5 ]
	then
		echo "Energy flux snapshots already merged"
	else
		echo "Merging energy flux snapshots"
		mpiexec -n $CORES python3 _energy_flux/ef_merge.py ef_snapshots
	fi
	echo "Plotting EF for z vs. t"
	mpiexec -n $CORES python3 _energy_flux/ef_plot_2d_series.py $LOC $NU $KA $N0 $NL ef_snapshots/*.h5
fi

echo "Done"
