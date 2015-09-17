#!/bin/sh
blockMesh
topoSet
setsToZones -noFlipMap
createBaffles -overwrite
