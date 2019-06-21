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

# On local pc
# VER = 1 and LOC=1
#	-> run the script, merge, plot, and create a gif
# VER = 2 and LOC=1
#	-> run the script

# On Niagara
# VER = 1 and LOC=0
#	-> merge, plot, and create a gif
# VER = 2 and LOC=0
#	-> run the script

# Define parameters
DN1=1.0E-1		# Rayleigh number
DN2=1.0E+0		# Prandtl number (~7 for water)
DN3=1.0E+3		# Reynolds number
DN4=1.0E+0		# N_0 (formerly, Richardson number)

RA=1e5
PR=7

# If VER = 1, then the code will merge, plot, and create a gif
# 	Check to see if the frames and snapshots folders exist
#	If so, remove them before running the dedalus script
if [ $VER -eq 1 ]
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
fi

# If running on local pc
if [ $LOC -eq 1 ]
then
	#echo 0 > /proc/sys/kernel/yama/ptrace_scope
	echo "Running Dedalus script for local pc"
	# mpiexec uses -n flag for number of processes to use
    mpiexec -n $CORES python3 current_code.py $LOC $DN1 $DN2 $DN3 $DN4
    echo ""
fi

# If running on Niagara
if [ $LOC -eq 0 ] && [ $VER -eq 2 ]
then
	echo "Running Dedalus script for Niagara"
	# mpirun uses -c, -n, --n, or -np for number of threads / cores
	#mpirun -c $CORES python3.6 current_code.py $LOC $DN1 $DN2 $DN3 $DN4
	# mpiexec uses -n flag for number of processes to use
    mpiexec -n $CORES python3.6 current_code.py $LOC $DN1 $DN2 $DN3 $DN4
	echo ""
fi

# If VER = 1, then run the rest of the code,
# 	but first check if snapshots folder was made
if [ $VER -eq 1 ] && [ -e snapshots ]
then
	echo "Merging snapshots"
	mpiexec -n $CORES python3 merge.py snapshots
	echo ""
	echo "Plotting 2d series"
	mpiexec -n $CORES python3 plot_2d_series.py snapshots/*.h5 --ND1=$DN1 --ND2=$DN2 --ND3=$DN3 --ND4=$DN4
	echo ""
	echo "Creating gif"
	python3 create_gif.py gifs/test.gif
fi

echo "Done"
