#!/bin/sh

if [ -z "$1" ] then
    cd $1
fi

cp system/createPatchDict.patch system/createPatchDict
createPatch -overwrite
rm -f *.obj
extrudeMesh
transformPoints -translate "(0 0 -0.01)"
cp system/createPatchDict.cyclic system/createPatchDict
createPatch -overwrite
rm -f *.obj
