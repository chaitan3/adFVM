#!/bin/sh
CASE=$1
NP=$2
echo $CASE $NP
sed -i "s/numberOfSubdomains[[:space:]]\+[0-9]\+;/numberOfSubdomains $NP;/g" $CASE/system/decomposeParDict
decomposePar -case $1 ${*:3}
mkdir $CASE/par-$2
mv $CASE/processor* $CASE/par-$2
