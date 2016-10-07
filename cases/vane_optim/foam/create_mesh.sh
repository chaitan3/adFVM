#!/bin/sh

fluentMeshToFoam ~/foam/vane/mesh/automate-mesh/mesh2d.msh
transformPoints -scale "(1 1 0.001)"
createPatch -overwrite
rm -f *.obj

