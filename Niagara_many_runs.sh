#!/bin/bash
# A bash script to run the Dedalus python code many times on Niagara
# run.sh takes in these arguments:
#	$ sh run.sh -n <name of particular run>
#				-c <cores>
#				-e <EF(1) or reproducing run(0)>
#				-i <number of interfaces in background profile>
#				-l <local(1) or Niagara(0)>
#				-v <version: what scripts to run>
#				-k <keep(1) or allow overwriting(0)>
#				-t <test parameter>

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
	echo "No number of cores specified, using CORES=40"
	CORES=40
fi

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
sh run.sh -c $CORES -v 1 -l 0 -e 1 -i 0 -k 1 -t 35

echo "Many runs script done"
