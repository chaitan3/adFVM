#!/usr/bin/python
import sys, os, copy
import numpy as np

from adFVM import config
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh

#config.hdf5 = True
case1, case2, time1, time2 = sys.argv[1:5]
if len(sys.argv) > 5:
    skipZ = True
else:
    skipZ = False
time1 = float(time1)
time2 = float(time2)

mesh1 = Mesh.create(case1)
mesh2 = Mesh.create(case2)

from scipy.spatial import cKDTree as KDTree
def mapNearest(centres1, centres2):
    if skipZ:
        centres1 = centres1[:,:2]
        centres2 = centres2[:,:2]
    tree = KDTree(centres1)
    indices = tree.query(centres2)[1]
    #print centres1.shape, centres2.shape, indices.shape
    return indices

def getPatchInfo(mesh, patchID):
    startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
    indices = mesh.neighbour[startFace:endFace]
    return startFace, endFace, indices

centres1 = mesh1.cellCentres[:mesh1.nInternalCells]
centres2 = mesh2.cellCentres[:mesh2.nInternalCells]
internalIndices = mapNearest(centres1, centres2)
patchIndices = {} 
for patchID in mesh1.boundary:
    startFace, endFace, indices1 = getPatchInfo(mesh1, patchID)
    centres1 = mesh1.faceCentres[startFace:endFace]
    startFace, endFace, indices2 = getPatchInfo(mesh2, patchID)
    centres2 = mesh2.faceCentres[startFace:endFace]
    patchIndices[patchID] = mapNearest(centres1, centres2)

#for field in mesh1.getFields(time1):
for field in ['U', 'T', 'p', 'rho', 'rhoU', 'rhoE']:
    print 'interpolating', field
    Field.setMesh(mesh1)
    with IOField.handle(time1):
        phi1 = IOField.read(field)
        phi1.partialComplete()
    dims = phi1.dimensions

    phi2 = np.zeros((mesh2.nCells, ) + dims)
    phi2 = IOField(field, phi2, dims)

    phi2.field[:mesh2.nInternalCells] = phi1.field[internalIndices]
    for patchID in phi1.boundary:
        startFace, endFace, indices1 = getPatchInfo(mesh1, patchID)
        startFace, endFace, indices2 = getPatchInfo(mesh2, patchID)
        phi2.field[indices2] = phi1.field[indices1][patchIndices[patchID]]
    phi2.boundary = copy.deepcopy(phi1.boundary)

    Field.setMesh(mesh2)
    with IOField.handle(time2):
        phi2.write()
    print
