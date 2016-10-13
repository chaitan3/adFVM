#!/bin/sh

cp system/createPatchDict.patch system/createPatchDict
createPatch -overwrite
rm -f *.obj
extrudeMesh
transformPoints -translate "(0 0 -0.01)"
cp system/createPatchDict.cyclic system/createPatchDict
createPatch -overwrite
rm -f *.obj
