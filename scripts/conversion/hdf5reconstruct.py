#!/usr/bin/python

import sys
import os
import h5py
import numpy as np
import copy

from adFVM import config
config.hdf5 = True

case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]

def getTimeDir(time, case):
    if time.is_integer():
        time = int(time)
        timeDir = '{0}/{1}'.format(case, time)
    else:
        timeDir = '{0}/{1:.11f}'.format(case, time)
    return timeDir

delta = 14
#delta = 5
if len(times) == 0:
    #times = [float(x[:-delta]) for x in os.listdir(case) if config.isfloat(x[:-delta]) and x.endswith('.hdf5')]
    times = [float(x[:-delta]) for x in os.listdir(case) if config.isfloat(x[:-delta]) and x.endswith('_parallel.hdf5')]
for index, time in enumerate(times):
    if time.is_integer():
        times[index] = int(time)
print('moving files')
try:
    path = case + '/mesh.hdf5'
    parallel_path = case + '/mesh_parallel.hdf5'
    if not os.path.exists(parallel_path):
        os.rename(path, parallel_path)
    #os.rename(parallel_path, path)
    for time in times:
        path = getTimeDir(time, case) + '.hdf5'
        parallel_path = getTimeDir(time, case) + '_parallel.hdf5'
        if not os.path.exists(parallel_path):
            os.rename(path, parallel_path)
        #os.rename(parallel_path, path)
except OSError:
    pass
#exit(1)

print 'reading hdf5 mesh'
mesh = h5py.File(case + '/mesh_parallel.hdf5', 'r')

parallelStart = mesh['parallel/start'][:]
parallelEnd = mesh['parallel/end'][:]
nProcs = parallelStart.shape[0]
cellsData = mesh['cells']
facesData = mesh['faces']
pointsData = mesh['points']
ownerData = mesh['owner']
neighbourData = mesh['neighbour']
faceSerial = facesData[:]
ownerSerial = -np.ones_like(ownerData)
neighbourSerial = ownerData[:]
cellSerial = cellsData[:]
pointSerial = pointsData[:]

boundaryGroup = mesh['boundary']
boundaryParallelStart = boundaryGroup['parallel/start'][:,0]
boundaryParallelEnd = boundaryGroup['parallel/end'][:,0]
boundaryData = boundaryGroup['values']
boundarySerial = []

nPoints = 0
nFaces = 0
cellAddressingAll = []
boundaryAddressing = []
boundaryProc = []
boundaryRange = {}
parallelInfo = parallelEnd-parallelStart

for proc in range(0, nProcs):
    cellAddressing = mesh['cellProcAddressing'][parallelStart[proc,7]:parallelEnd[proc,7]]
    faceAddressing = np.abs(mesh['faceProcAddressing'][parallelStart[proc,6]:parallelEnd[proc,6]])-1
    pointAddressing = mesh['pointProcAddressing'][parallelStart[proc,5]:parallelEnd[proc,5]]

    points = pointsData[parallelStart[proc,1]:parallelEnd[proc,1]]
    faces = facesData[parallelStart[proc,0]:parallelEnd[proc,0]]
    owner = ownerData[parallelStart[proc,2]:parallelEnd[proc,2]]
    neighbour = neighbourData[parallelStart[proc,3]:parallelEnd[proc,3]]
    cells = cellsData[parallelStart[proc,4]:parallelEnd[proc,4]]
    cells = pointAddressing[cells]
    faces[:,1:] = pointAddressing[faces[:,1:]]
    owner = cellAddressing[owner]
    neighbour = cellAddressing[neighbour]
    nFaces = max(nFaces, faceAddressing.max() + 1)
    nPoints = max(nPoints, pointAddressing.max() + 1)
    pointSerial[pointAddressing] = points
    faceSerial[faceAddressing] = faces
    cellSerial[cellAddressing] = cells
    valOwnerSerial = ownerSerial[faceAddressing]
    negOwnerSerial = (valOwnerSerial == -1)
    posOwnerSerial = (valOwnerSerial != -1)
    ownerSerial[faceAddressing[negOwnerSerial]] = owner[negOwnerSerial]
    neighbourSerial[faceAddressing[posOwnerSerial]] = owner[posOwnerSerial]
    neighbourSerial[faceAddressing[:neighbour.shape[0]]] = neighbour
    cellAddressingAll.append(cellAddressing)

    boundary = {}
    for patchID, key, value in boundaryData[boundaryParallelStart[proc]:boundaryParallelEnd[proc]]:
        if patchID not in boundary:
            boundary[patchID] = {}
        if key == 'startFace' or key == 'nFaces':
            boundary[patchID][key] = int(value)
        else:
            boundary[patchID][key] = value
    boundary = {k:v for k,v in boundary.iteritems() if v['type'] != 'processor'}
    boundaryAddressing.append({})
    for patchID in boundary:
        patch = boundary[patchID]
        patchFaces = np.arange(patch['startFace'], patch['startFace']+patch['nFaces'])
        if len(patchFaces) == 0:
            continue
        patchFaces = faceAddressing[patchFaces]
        if patch['type'] != 'processorCyclic':
            boundaryAddressing[-1][patchID] = patchFaces
        else:
            patchID = patch['referPatch']
        if patchID not in boundaryRange:
            boundaryRange[patchID] = (2**62,0)
        currRange = boundaryRange[patchID]
        boundaryRange[patchID] = min(currRange[0], patchFaces.min()), max(currRange[1], patchFaces.max() + 1)
    boundary = {k:v for k,v in boundary.iteritems() if v['type'] != 'processorCyclic'}
    boundaryProc.append(boundary)
            
cellAddressing = np.concatenate(cellAddressingAll)
pointSerial = pointSerial[:nPoints]
faceSerial = faceSerial[:nFaces]
nInternalFaces = nFaces

for patchID in boundaryProc[0]:
    faceRange = boundaryRange[patchID]
    nInternalFaces = min(faceRange[0], nInternalFaces)
    for key, value in boundaryProc[0][patchID].items():
        if key == 'startFace':
            value = str(faceRange[0])
        elif key == 'nFaces':
            value = str(faceRange[1]-faceRange[0])
        boundarySerial.append([patchID, key, value])
ownerSerial = ownerSerial[:nFaces]
neighbourSerial = neighbourSerial[:nInternalFaces]
mesh.close()

path = case + '/mesh.hdf5'
if not os.path.exists(path):
    print('writing serial hdf5 mesh')
    meshSerial = h5py.File(path, 'w')
    meshSerial.create_dataset('cells', data=cellSerial)
    meshSerial.create_dataset('faces', data=faceSerial)
    meshSerial.create_dataset('owner', data=ownerSerial)
    meshSerial.create_dataset('neighbour', data=neighbourSerial)
    meshSerial.create_dataset('points', data=pointSerial)
    parallelGroup = meshSerial.create_group('parallel')
    parallelGroup.create_dataset('start', data=np.zeros((1,5), np.int64))
    parallelGroup.create_dataset('end', data=np.array([[faceSerial.shape[0],pointSerial.shape[0],ownerSerial.shape[0],neighbourSerial.shape[0],cellSerial.shape[0]]], np.int64))

    boundaryGroup = meshSerial.create_group('boundary')
    boundaryGroup.create_dataset('values', data=np.array(boundarySerial, dtype='S100'))
    parallelGroup = boundaryGroup.create_group('parallel')
    boundaryGroup.create_group('fields')
    parallelGroup.create_dataset('start', data=np.zeros((1,1), np.int64))
    parallelGroup.create_dataset('end', data=np.array([[len(boundarySerial)]], np.int64))
    meshSerial.close()

#exit(1)
for time in times:
    timeDir = getTimeDir(time, case)
    path = timeDir + '.hdf5'
    if os.path.exists(path):
        continue
    field = h5py.File(timeDir + '_parallel.hdf5', 'r')
    time = str(time)
    print 'reading hdf5 fields ' + time
    fieldSerial = h5py.File(path, 'w')
    for name in field.keys():
        if name == 'mesh':
            continue
        if name == 'parallel':
            continue
        #print 'reading field ' + name
        fieldData = field[name]['field'][:]
        boundaryData = field[name]['boundary'][:]
        parallelStart = field[name]['parallel/start']
        parallelEnd = field[name]['parallel/end']
        data = []
        for proc in range(0, nProcs):
            start = parallelStart[proc,0]
            data.append(fieldData[start:start+parallelInfo[proc, 4]])
        data = np.vstack(data)
        dims = (data.shape[0] + nFaces - nInternalFaces,) + data.shape[1:]
        dataSerial = np.zeros(dims, data.dtype)
        dataSerial[cellAddressing] = data

        boundary = []
        for patchID in boundaryProc[0]:
            for proc in range(0, nProcs):
                if boundaryProc[proc][patchID]['nFaces'] > 0:
                    start, end = parallelStart[proc,1], parallelEnd[proc,1]
                    for patchID2, key, value in boundaryData[start:end]:
                        if patchID2 == patchID:
                            boundary.append([patchID, key, value])
                    break
        for proc in range(0, nProcs):
            for patchID, addressing in boundaryAddressing[proc].items():
                patch = boundaryProc[proc][patchID]
                cellStartFace = parallelStart[proc,0] + parallelInfo[proc,4] - parallelInfo[proc,3] + patch['startFace']
                #if patchID == 'pressure' and patch['nFaces'] > 0:
                #    import pdb;pdb.set_trace()
                cellEndFace = cellStartFace + patch['nFaces']
                dataSerial[addressing-nInternalFaces+data.shape[0]] = fieldData[cellStartFace:cellEndFace]

        #print('writing serial field ' + name)
        fieldGroup = fieldSerial.create_group(name)
        fieldGroup.create_dataset('field', data=dataSerial)
        fieldGroup.create_dataset('boundary', data=np.array(boundary, dtype='S100'))
        parallelGroup = fieldGroup.create_group('parallel')
        parallelGroup.create_dataset('start', data=np.zeros((1,2), np.int64))
        parallelGroup.create_dataset('end', data=np.array([[dataSerial.shape[0],len(boundary)]], np.int64))

    fieldSerial.close()
    os.remove(timeDir + '_parallel.hdf5')
