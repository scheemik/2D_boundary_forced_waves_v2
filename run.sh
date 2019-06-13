#!/bin/bash
# A bash script to run the Dedalus python code
# Optionally takes in arguments:
#	$ sh run.sh -v <version_number>

while getopts v:l: option
do
	case "${option}"
		in
		v) VER=${OPTARG};;
		l) LOC=${OPTARG};;
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
DN2=7.0E+0		# Prandtl number
DN3=1.0E-2		# Reynolds number
DN4=0.0E+1		# Richardson number

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
    mpiexec -n 2 python3 current_code.py $LOC $DN1 $DN2 $DN3 $DN4
    echo ""
fi

# If running on Niagara
if [ $LOC -eq 0 ] && [ $VER -eq 2 ]
then
	echo "Running Dedalus script for Niagara"
	mpirun python3.6 current_code.py $LOC $DN1 $DN2 $DN3 $DN4
	echo ""
fi

# If VER = 1, then run the rest of the code,
# 	but first check if snapshots folder was made
if [ $VER -eq 1 ] && [ -e snapshots ]
then
	echo "Merging snapshots"
	python3 merge.py snapshots
	echo ""
	echo "Plotting 2d series"
	python3 plot_2d_series.py snapshots/*.h5 --ND1=$DN1 --ND2=$DN2 --ND3=$DN3 --ND4=$DN4
	echo ""
	echo "Creating gif"
	python3 create_gif.py gifs/test.gif
fi

echo "Done"
