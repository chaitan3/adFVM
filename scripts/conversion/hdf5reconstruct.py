#!/usr/bin/python2

import sys
import os
import h5py
import numpy as np
import copy

from adFVM import config
config.hdf5 = True

case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]
if len(times) == 0:
    times = [float(x[:-5]) for x in os.listdir(case) if config.isfloat(x[:-5]) and x.endswith('.hdf5')]
for index, time in enumerate(times):
    if time.is_integer():
        times[index] = int(time)

print('moving files')
try:
    path = case + '/mesh.hdf5'
    parallel_path = case + '/mesh_parallel.hdf5'
    if not os.path.exists(parallel_path):
        os.rename(path, parallel_path)
    for time in times:
        path = case + '/{}.hdf5'.format(time)
        parallel_path = case + '/{}_parallel.hdf5'.format(time)
        if not os.path.exists(parallel_path):
            os.rename(path, parallel_path)
except OSError:
    pass

print 'reading hdf5 mesh'
mesh = h5py.File(case + '/mesh_parallel.hdf5', 'r')

parallelStart = mesh['parallel/start']
parallelEnd = mesh['parallel/end']
nProcs = parallelStart.shape[0]
cellsData = mesh['cells']
facesData = mesh['faces']
pointsData = mesh['points']
faceSerial = facesData[:]
cellSerial = cellsData[:]
pointSerial = pointsData[:]

boundaryGroup = mesh['boundary']
boundaryParallelStart = boundaryGroup['parallel/start'][:,0]
boundaryParallelEnd = boundaryGroup['parallel/end'][:,0]
boundaryData = boundaryGroup['values']

nPoints = 0
nFaces = 0
cellAddressingAll = []
nCells = []
boundarySerial = {}

for proc in range(0, nProcs):
    cellAddressing = mesh['cellProcAddressing'][parallelStart[proc,7]:parallelEnd[proc,7]]
    faceAddressing = mesh['faceProcAddressing'][parallelStart[proc,6]:parallelEnd[proc,6]]
    pointAddressing = mesh['pointProcAddressing'][parallelStart[proc,5]:parallelEnd[proc,5]]

    points = pointsData[parallelStart[proc,1]:parallelEnd[proc,1]]
    faces = facesData[parallelStart[proc,0],parallelEnd[proc,0]]
    cells = cellsData[parallelStart[proc,4]:parallelEnd[proc,4]]
    cells = pointAddressing[cells]
    faces = pointAddressing[faces]
    nCells.append(cells.shape[0])
    nFaces = max(nFaces, faceAddresing.max() + 1)
    nPoints = max(nPoints, pointAddressing.max() + 1)
    pointSerial[pointAddressing] = points
    faceSerial[faceAddressing] = faces
    cellSerial[cellAddressing] = cells
    cellAddressingAll.append(cellAddressing)

    boundary = {}
    for patchID, key, value in boundaryData[boundaryParallelStart[proc]:boundaryParallelEnd[proc]]
        if not patchID.startswith('procBoundary'):
            if patchID not in boundary:
                boundary[patchID] = {}
            if key == 'startFace' or key == 'nFaces':
                boundary[patchID][key] = int(value)
            else:
                boundary[patchID][key] = value
    if proc == 0:
        boundarySerial = copy.deepcopy(boundary)
    for patchID in boundarySerial:
        patchSerial = boundarySerial[patchID]
        patch = boundary[patchID]
        patchFaces = np.arange(patch['startFace'], patch['startFace']+patch['nFaces'])
        patchFaces = faceA
        patchSerial['startFace'] = min(patchSerial['startFace'], patch['startFace'])
            
cellAddressing = np.concatenate(cellAddressingAll)
pointSerial = pointSerial[:nPoints]
faceSerial = faceSerial[:nFaces]


mesh.close()

path = case + '/mesh.hdf5'
if not os.path.exists(path):
    print('writing serial hdf5 mesh')
    meshSerial = h5py.File(path, 'w')
    meshSerial.create_dataset('cells', data=cellSerial)
    meshSerial.create_dataset('faces', data=faceSerial)
    meshSerial.create_dataset('points', data=pointSerial)
    parallelGroup = meshSerial.create_group('parallel')
    parallelGroup.create_dataset('start', data=np.zeros((1,5), np.int64))
    parallelGroup.create_dataset('end', data=np.array([[0,pointSerial.shape[0],0,0,cellSerial.shape[0]]], np.int64))

    boundaryGroup = meshSerial.create_group('boundary')
    parallelGroup = boundaryGroup.create_group('parallel')
    parallelGroup.create_dataset('start', data=np.zeros((1,1), np.int64))
    parallelGroup.create_dataset('end', data=np.array([[len(boundaryData)]], np.int64))
    meshSerial.close()

for time in times:
    time = str(time)
    print 'reading hdf5 fields ' + time
    path = case + '/{}.hdf5'.format(time)
    if os.path.exists(path):
        continue
    field = h5py.File(case + '/{}_parallel.hdf5'.format(time), 'r')
    fieldSerial = h5py.File(path, 'w')
    for name in field.keys():
        if name == 'mesh':
            continue
        if name == 'parallel':
            continue
        print 'reading field ' + name
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

        print('writing serial field ' + name)
        fieldGroup = fieldSerial.create_group(name)
        fieldGroup.create_dataset('field', data=dataSerial)
        parallelGroup = fieldGroup.create_group('parallel')
        parallelGroup.create_dataset('start', data=np.zeros((1,2), np.int64))
        parallelGroup.create_dataset('end', data=np.array([[cellSerial.shape[0],0]], np.int64))

    fieldSerial.close()
