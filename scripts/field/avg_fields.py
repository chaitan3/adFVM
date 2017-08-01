#!/usr/bin/python2
import sys
import os
import numpy as np

from adFVM import config, BCs
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh
from adFVM.parallel import pprint

case = sys.argv[1]
fields = sys.argv[2:]
mesh = Mesh.create(case)
Field.setMesh(mesh)
meshO = mesh.origMesh

times = mesh.getTimes()
#times = filter(lambda x: x > 1.002, times)
pprint(times)

config.hdf5 = True
nLayers = 200
#nLayers = 1

for field in fields:
    # time avg: no dt
    avg = 0.
    for time in times:
        with IOField.handle(time):
            phi = IOField.read(field)
        phi.partialComplete()
        avg += phi.field
    avg /= len(times)

    # spanwise avg: structured
    nDims = avg.shape[1]
    #nLayers = 1
    def average(start, end):
        nCellsPerLayer = (end-start)/nLayers
        spanAvg = avg[start:end].reshape((nLayers, nCellsPerLayer, nDims)).sum(axis=0)/nLayers
        spanAvg = np.tile(spanAvg, (nLayers,1))
        avg[start:end] = spanAvg

    average(0, meshO.nInternalCells)
    for patchID in mesh.localPatches:
        patch = meshO.boundary[patchID]
        if patch['type'] in BCs.valuePatches:
            cellStartFace, cellEndFace, _ = meshO.getPatchCellRange(patchID)
            average(cellStartFace, cellEndFace)

    phi.name = field + '_avg'
    phi.field = avg
    with IOField.handle(times[0]):
        phi.write()
