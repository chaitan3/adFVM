#!/bin/sh

CASES=`pwd`/../cases

blockMesh -case $CASES/convection
#blockMesh -case $CASES/burgers
#blockMesh -case $CASES/shockTube

blockMesh -case $CASES/forwardStep
#cd $CASES/cylinder && ./create_mesh.sh
