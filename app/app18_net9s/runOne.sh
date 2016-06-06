#!/bin/bash
strRunName=$1_$2_$3
cmd="th $1 $2 $3 > result/$strRunName.out 2> result/$strRunName.err"
eval "$cmd" &
disown -h %1
