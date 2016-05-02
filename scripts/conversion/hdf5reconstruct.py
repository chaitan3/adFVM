#!/usr/bin/python2

import sys
import os
import h5py
import numpy as np

from mesh import Mesh

case = sys.argv[1]
times = sys.argv[2:]

def getAddressing(proc):
    mesh = Mesh()

    foamCase = case + 'processor' + str(proc) + '/constant/polyMesh/'
    print 'processing ' + foamCase
    cellAddressing = mesh.readFoamFile(foamCase + 'cellProcAddressing', np.int32)
    pointAddressing = mesh.readFoamFile(foamCase + 'pointProcAddressing', np.int32)

    return cellAddressing, pointAddressing

print 'reading hdf5 mesh'
mesh = h5py.File(case + '/mesh.hdf5', 'r')
parallelStart = mesh['parallel/start']
parallelEnd = mesh['parallel/end']
nProcs = parallelStart.shape[0]
cellsData = mesh['cells']
pointsData = mesh['points']


cellSerial = cellsData[:]
pointSerial = pointsData[:]

nPoints = 0
cellAddressingAll = []
nCells = []
for proc in range(0, nProcs):
    cellAddressing, pointAddressing = getAddressing(proc)

    points = pointsData[parallelStart[proc,1]:parallelEnd[proc,1]]
    cells = cellsData[parallelStart[proc,5]:parallelEnd[proc,5]]
    cells = pointAddressing[cells]
    nCells.append(cells.shape[0])
    
    nPoints = max(nPoints, pointAddressing.max() + 1)
    pointSerial[pointAddressing] = points
    cellSerial[cellAddressing] = cells
    cellAddressingAll.append(cellAddressing)
cellAddressing = np.concatenate(cellAddressingAll)
pointSerial = pointSerial[:nPoints]
mesh.close()

meshSerial = h5py.File(case + '/mesh_serial.hdf5', 'w')
meshSerial.create_dataset('cells', data=cellSerial)
meshSerial.create_dataset('points', data=pointSerial)
parallelGroup = meshSerial.create_group('parallel')
parallelGroup.create_dataset('start', data=np.zeros((1,6), np.int64))
parallelGroup.create_dataset('end', data=np.array([[0,pointSerial.shape[0],0,0,0,cellSerial.shape[0]]], np.int64))
meshSerial.close()

for time in times:
    print 'reading hdf5 fields ' + time
    field = h5py.File(case + '/{}.hdf5'.format(time), 'r')
    fieldSerial = h5py.File(case + '/{}_serial.hdf5'.format(time), 'w')
    for name in field.keys():
        print 'reading field ' + name
        if name == 'parallel':
            continue
        fieldData = field[name]['field'][:]
        parallelStart = field[name]['parallel/start']
        parallelEnd = field[name]['parallel/end']
        data = []
        for proc in range(0, nProcs):
            start = parallelStart[proc,0]
            data.append(fieldData[start:start+nCells[proc]])
        data = np.vstack(data)
        dataSerial = np.zeros_like(data)
        dataSerial[cellAddressing] = data

        fieldGroup = fieldSerial.create_group(name)
        fieldGroup.create_dataset('field', data=dataSerial)
        parallelGroup = fieldGroup.create_group('parallel')
        parallelGroup.create_dataset('start', data=np.zeros((1,2), np.int64))
        parallelGroup.create_dataset('end', data=np.array([[cellSerial.shape[0],0]], np.int64))

    fieldSerial.close()
