#!/bin/sh
DIR=$1
TIME=$2
cp $DIR/constant/polyMesh/boundary.cyclic $DIR/constant/polyMesh/boundary
find $DIR/$TIME -type f -exec sed -i "s/slidingPeriodic1D/cyclic/g" {} \;
cp $DIR/$TIME/polyMesh/boundary $DIR/$TIME/polyMesh/boundary.sliding
cp $DIR/$TIME/polyMesh/boundary.cyclic $DIR/$TIME/polyMesh/boundary
