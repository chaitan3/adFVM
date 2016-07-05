from field import Field, IOField
from mesh import Mesh
import sys
import os
import numpy as np
import config

case, field = sys.argv[1:3]
#times = sys.argv[3:]
mesh = Mesh.create(case)
Field.setMesh(mesh)
meshO = mesh.origMesh

# time avg: no dt
times = mesh.getTimes()
with IOField.handle(times[0]):
    phiAvg = IOField.read(field + '_avg')
    phiAvg.partialComplete()

std = 0.
for time in times:
    with IOField.handle(time):
        phi = IOField.read(field)
        phi.partialComplete()
    std += (phiAvg.field-phi.field)**2

# spanwise avg: structured
nLayers = 200
nDims = std.shape[1]
#nLayers = 1
def average(start, end):
    nCellsPerLayer = (end-start)/nLayers
    spanAvg = std[start:end].reshape((nLayers, nCellsPerLayer, nDims)).sum(axis=0)/nLayers
    spanAvg = np.tile(spanAvg, (nLayers,1))
    std[start:end] = spanAvg

average(0, meshO.nInternalCells)
for patchID in mesh.localPatches:
    patch = meshO.boundary[patchID]
    if patch['type'] in BCs.valuePatches:
        cellStartFace = patch['startFace']-meshO.nInternalFaces + meshO.nInternalCells
        cellEndFace = cellStartFace + patch['nFaces']
        average(cellStartFace, cellEndFace)


std = np.sqrt(std/len(times))

phi.name = field + '_std'
phi.field = std
with IOField.handle(times[0])
    phi.write()
