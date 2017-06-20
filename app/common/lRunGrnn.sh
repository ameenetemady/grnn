#!/bin/bash

nMin=${1:-1}
nMax=${2:-20}

mySRun() {
	strFilename=".sbatch_scripts/grnn_$1_$2_$3.sh"
	rm -f $strFilename

	echo "#!/bin/bash" >> $strFilename
	echo "th ../common/run_grnn_one.lua $1 $2 $3" >> $strFilename
	chmod u+x $strFilename

	sbatch -n 1 -N 1 -t 120 --job-name=$strFilename $strFilename
}

distList="0.3 0.2 0.15 0.1 0.05 0.02 0.01 0.005 0.002 0.001 0.000"
#distList="0.3" # for test
mkdir -p .sbatch_scripts

for i in $distList
do
	for j in $(seq $nMin $nMax)
	do
		mySRun true $i $j
		mySRun false $i $j
	done
done


