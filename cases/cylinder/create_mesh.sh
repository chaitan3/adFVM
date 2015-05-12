#!/bin/sh
blockMesh
mirrorMesh
createPatch -overwrite
rm -f *.obj
