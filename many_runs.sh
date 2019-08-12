#!/bin/bash
# A bash script to run the Dedalus python code many times
# run.sh takes in these arguments:
#	$ sh run.sh -n <name of particular run>
#				-c <cores>
#				-e <EF(1) or reproducing run(0)>
#				-i <number of interfaces in background profile>
#				-l <local(1) or Niagara(0)>
#				-v <version: what scripts to run>
#				-k <keep(1) or allow overwriting(0)>
#				-t <test parameter>

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

echo 'Test parameter currently set to nz'
sh run.sh -v 0 -e 1 -i 1 -k 1 -t 512
sh run.sh -v 0 -e 1 -i 1 -k 1 -t 524
sh run.sh -v 0 -e 1 -i 1 -k 1 -t 532
sh run.sh -v 0 -e 1 -i 1 -k 1 -t 548

echo "Many runs script done"
