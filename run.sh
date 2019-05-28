#!/bin/bash
# A bash script to run the Dedalus python code

# which version?
VER=2
# Define parameters
A1=1.0
B2=1.1
C3=1.2
D4=1.4

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
  python3 current_code.py $A1 $B2 $C3 $D4
  echo ""
  # check if snapshots folder was made
  if [ -e snapshots ]
  then
    echo "Merging snapshots"
    python3 merge.py snapshots
    echo ""
    echo "Plotting 2d series"
    python3 plot_2d_series.py snapshots/*.h5 --ND1=$A1 --ND2=$B2 --ND3=$C3 --ND4=$D4
    echo ""
    echo "Creating gif"
    python3 create_gif.py gifs/test.gif
  fi
elif [ $VER -eq 2 ]
then
  echo "Running Dedalus script"
  python3 current_code.py $A1 $B2 $C3 $D4
  echo ""
fi

echo "Done"
