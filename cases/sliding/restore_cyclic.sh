#!/bin/sh
cp constant/polyMesh/boundary.cyclic constant/polyMesh/boundary
find 1 -type f -exec sed -i "s/slidingPeriodic1D/cyclic/g" {} \;
