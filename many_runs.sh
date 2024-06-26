#!/bin/bash
# A bash script to run the Dedalus python code many times

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

echo 'Test parameter currently set to forcing slope'
sh run.sh -v 0 -e 0 -i 2 -k 1 -t 35
sh run.sh -v 0 -e 0 -i 1 -k 1 -t 35

echo "Many runs script done"
