#!/bin/bash
CASE=$1
NP=$2
ARGS="${@:3}"
echo $CASE $NP
sed -i "s/numberOfSubdomains[[:space:]]\+[0-9]\+;/numberOfSubdomains $NP;/g" $CASE/system/decomposeParDict
decomposePar -case $CASE $ARGS
mkdir $CASE/par-$NP
mv $CASE/processor* $CASE/par-$NP
