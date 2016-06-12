#!/bin/bash
. /home/user/eetemame/torch/install/bin/torch-activate
cd /home/user/eetemame/mygithub/grnn/app/app18_net9s
strRunName=$1_$2_$3
cmd="th $1 $2 $3 > result/$strRunName.out 2> result/$strRunName.err"
echo "running" $cmd
eval "$cmd" 
echo "done"
