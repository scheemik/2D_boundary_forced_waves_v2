#!/bin/bash
# A bash script to run the Dedalus python code

# which version? 1=b, 2=T, 3=S
VER=1
# Define parameters
RA=1e5
PR=7
TAU=1.1e-2
RP=1e-2

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
  #rm gifs/test.gif
  echo "Running Rayleigh Benard script"
  python3 2D_bfw_code.py $RA $PR
  echo ""
  echo "Merging snapshots"
  python3 merge.py snapshots
  echo ""
  echo "Plotting 2d series"
  python3 plot_2d_series.py snapshots/*.h5 --rayleigh=$RA --prandtl=$PR
  echo ""
  echo "Creating gif"
  python3 create_gif.py gifs/test.gif
elif [ $VER -eq 2 ]
then
  rm T_RB.gif
  echo "Running Rayleigh Benard script"
  python3 T_rb_code.py $PR $TAU $RP
  echo ""
  echo "Merging snapshots"
  python3 merge.py snapshots
  echo ""
  echo "Plotting 2d series"
  python3 T_plot_2d_series.py snapshots/*.h5 --prandtl=$PR --diffusivity_r=$TAU --density_r=$RP
  echo ""
  echo "Creating gif"
  python3 create_gif.py T_RB.gif
fi

echo "Done"
