#!/bin/bash
DIR=$(dirname "${BASH_SOURCE[0]}")
source /opt/openfoam6/etc/bashrc

CASES=$DIR/../cases

blockMesh -case $CASES/convection
blockMesh -case $CASES/forwardStep

#blockMesh -case $CASES/burgers
#blockMesh -case $CASES/shockTube
#cd $CASES/cylinder && ./create_mesh.sh
