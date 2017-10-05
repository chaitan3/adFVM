#!/usr/bin/python2
import sys
import os
import numpy as np

from adFVM import config, BCs
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh
from adFVM.parallel import pprint

#config.hdf5 = True
case = sys.argv[1]
fields = sys.argv[2:]
mesh = Mesh.create(case)
Field.setMesh(mesh)

times = mesh.getTimes()
times = filter(lambda x: x > 3.00049, times)
pprint(times)

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
        spanAvg = avg[start:end].reshape((nCellsPerLayer, nLayers, nDims)).sum(axis=1,keepdims=1)/nLayers
        spanAvg = np.tile(spanAvg, (1,nLayers,1)).reshape((end-start, nDims))
        avg[start:end] = spanAvg

    average(0, mesh.nInternalCells)
    for patchID in mesh.localPatches:
        patch = mesh.boundary[patchID]
        if patch['type'] in BCs.valuePatches:
            cellStartFace, cellEndFace, _ = mesh.getPatchCellRange(patchID)
            average(cellStartFace, cellEndFace)

    phi.name = field + '_avg'
    phi.field = avg
    with IOField.handle(times[0]):
        phi.write()
