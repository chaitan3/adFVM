from field import Field, IOField
from mesh import Mesh
import sys
import os
import numpy as np
import config

def isfloat(s):
    try:
        f = float(s)
        return True
    except ValueError:
        return False

case, field = sys.argv[1:3]
#times = sys.argv[3:]
times = sorted([float(x) for x in os.listdir(case) if isfloat(x) ])
mesh = Mesh.create(case)
Field.setMesh(mesh)
meshO = mesh.origMesh

# time avg: no dt
avg = 0.
for time in times:
    IOField.openHandle(time)
    phi = IOField.read(field)
    IOField.closeHandle()
    phi.partialComplete()
    avg += phi.field
avg /= len(times)

# spanwise avg: structured
nLayers = 200
def average(start, end):
    nCellsPerLayer = (end-start)/nLayers
    spanAvg = avg[start:end].reshape((nLayers, nCellsPerLayer)).sum(axis=0)/nLayers
    spanAvg = np.tile(spanAvg, (nLayers,1)).reshape(-1,1)
    avg[start:end] = spanAvg

average(0, meshO.nInternalCells)
for patchID in mesh.localPatches:
    patch = meshO.boundary[patchID]
    if patch['type'] != 'cyclic':
        cellStartFace = patch['startFace']-meshO.nInternalFaces + meshO.nInternalCells
        cellEndFace = cellStartFace + patch['nFaces']
        average(cellStartFace, cellEndFace)

phi.name = field + '_avg'
phi.field = avg
IOField.openHandle(times[0])
phi.write()
IOField.closeHandle()
