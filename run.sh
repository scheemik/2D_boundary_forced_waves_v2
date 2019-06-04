#!/bin/bash
# A bash script to run the Dedalus python code

# which version?
VER=1
# Define parameters
DN1=1.0E-1		# Euler number
DN2=7.0E+0		# Prandtl number
DN3=1.0E-2		# Reynolds number
DN4=1.0E+1		# Richardson number

RA=1e5
PR=7

# Check to see if the frames and snapshots folders exist
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

if [ $VER -eq 1 ]
then
  echo "Running Dedalus script"
  #mpiexec -n 2 python3 current_code.py $ND1 $ND2 $ND3 $ND4
  python3 current_code.py $DN1 $DN2 $DN3 $DN4
  echo ""
  # check if snapshots folder was made
  if [ -e snapshots ]
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
elif [ $VER -eq 2 ]
then
  echo "Running Dedalus script"
  python3 current_code.py $DN1 $DN2 $DN3 $DN4
  echo ""
fi

echo "Done"
