#!/usr/bin/python2
import sys
import os
import glob
import h5py
import numpy as np
import re

import parallel
import config

from mesh import Mesh
from field import IOField
import BCs

case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]
processorDirs = sorted(glob.glob(case + 'processor*'))

meshFile = h5py.File(case + '/mesh.hdf5', 'w')

def meshWriteHDF5(self, rank, maxRank):
    mesh = self
    print('writing hdf5 mesh')

    boundary = []
    for patchID in mesh.boundary.keys():
        for key, value in mesh.boundary[patchID].iteritems():
            boundary.append([patchID, key, str(value)])
    boundary = np.array(boundary, dtype='S100')
    faces, points, owner, neighbour = self.faces, self.points, mesh.owner, mesh.neighbour
    if mesh.nInternalFaces > 0:
        neighbour = neighbour[:mesh.nInternalFaces]

    parallelInfo = np.array([faces.shape[0], points.shape[0], \
                             owner.shape[0], neighbour.shape[0],
                             boundary.shape[0]])

    if rank == 0:
        parallelGroup = meshFile.create_group('parallel')
        parallelStartData = parallelGroup.create_dataset('start', (maxRank, len(parallelInfo)), np.int64)
        parallelEndData = parallelGroup.create_dataset('end', (maxRank, len(parallelInfo)), np.int64)
        parallelStart = np.zeros_like(parallelInfo)
        parallelEnd = parallelInfo

        facesData = meshFile.create_dataset('faces', (parallelInfo[0],) + faces.shape[1:], faces.dtype, maxshape=(None,) + faces.shape[1:])
        pointsData = meshFile.create_dataset('points', (parallelInfo[1],) + points.shape[1:], np.float64, maxshape=(None,) + points.shape[1:])
        ownerData = meshFile.create_dataset('owner', (parallelInfo[2],) + owner.shape[1:], owner.dtype, maxshape=(None,) + owner.shape[1:])
        neighbourData = meshFile.create_dataset('neighbour', (parallelInfo[3],) + neighbour.shape[1:], neighbour.dtype, maxshape=(None,) + neighbour.shape[1:])
        boundaryData = meshFile.create_dataset('boundary', (parallelInfo[4], 3), 'S100', maxshape=(None,3)) 
    else:
        parallelStartData = meshFile['parallel/start']
        parallelEndData = meshFile['parallel/end']
        parallelStart = parallelEndData[rank-1]
        parallelEnd = parallelStart + parallelInfo

        facesData = meshFile['faces']
        pointsData = meshFile['points']
        ownerData = meshFile['owner']
        neighbourData = meshFile['neighbour']
        boundaryData = meshFile['boundary']
        #facesData.resize(facesData.shape[0] + faces.shape[0], axis=0)
        #pointsData.resize(pointsData.shape[0] + points.shape[0], axis=0)
        #ownerData.resize(ownerData.shape[0] + owner.shape[0], axis=0)
        #neighbourData.resize(neighbourData.shape[0] + neighbour.shape[0], axis=0)
        #boundaryData.resize(boundaryData.shape[0] + boundary.shape[0], axis=0)
    
    parallelStartData[rank] = parallelStart
    parallelEndData[rank] = parallelEnd

    facesData[parallelStart[0]:parallelEnd[0]] = faces
    pointsData[parallelStart[1]:parallelEnd[1]] = points.astype(np.float64)
    ownerData[parallelStart[2]:parallelEnd[2]] = owner
    neighbourData[parallelStart[3]:parallelEnd[3]] = neighbour

    boundaryData[parallelStart[4]:parallelEnd[4]] = boundary

def fieldWriteHDF5(self, time, rank, maxRank):
    # mesh values required outside theano
    print('writing hdf5 field {0}, time {1}'.format(self.name, times[time]))

    boundary = []
    for patchID in self.boundary.keys():
        #print rank, self.name, patchID
        patch = self.boundary[patchID]
        for key, value in patch.iteritems():
            if not (key == 'value' and patch['type'] in BCs.valuePatches):
                boundary.append([patchID, key, str(value)])
    boundary = np.array(boundary, dtype='S100')

    # fetch processor information
    field = self.field

    global fieldsFiles
    fieldsFile = fieldsFiles[time]
    #fieldGroup.create_dataset('dimensions', data=self.dimensions)

    parallelInfo = np.array([field.shape[0], boundary.shape[0]])

    if rank == 0:
        fieldGroup = fieldsFile.create_group(self.name)
        parallelGroup = fieldGroup.create_group('parallel')
        parallelStartData = parallelGroup.create_dataset('start', (maxRank, len(parallelInfo)), np.int64)
        parallelEndData = parallelGroup.create_dataset('end', (maxRank, len(parallelInfo)), np.int64)
        parallelStart = np.zeros_like(parallelInfo)
        parallelEnd = parallelInfo

        fieldData = fieldGroup.create_dataset('field', (parallelInfo[0],) + self.dimensions, np.float64, maxshape=(None,) + self.dimensions)
        boundaryData = fieldGroup.create_dataset('boundary', (parallelInfo[1], 3), 'S100', maxshape=(None,3)) 
    else:
        fieldGroup = fieldsFile[self.name]
        parallelStartData = fieldGroup['parallel/start']
        parallelEndData = fieldGroup['parallel/end']
        parallelStart = parallelEndData[rank-1]
        parallelEnd = parallelStart + parallelInfo

        fieldData = fieldGroup['field']
        boundaryData = fieldGroup['boundary']

    parallelStartData[rank] = parallelStart
    parallelEndData[rank] = parallelEnd

    fieldData[parallelStart[0]:parallelEnd[0]] = field.astype(np.float64)

    boundaryData[parallelStart[1]:parallelEnd[1]] = boundary

fieldsFiles = []
for time in times:
    if time.is_integer():
        time = int(time)
    fieldsFiles.append(h5py.File(case + '/' + str(time) + '.hdf5', 'w'))

for rank, processor in enumerate(processorDirs):
    config.hdf5 = False
    print processor
    mesh = Mesh()
    mesh.readFoam(processor, 'constant')
    mesh.populateSizes()
    mesh.origMesh = mesh
    meshWriteHDF5(mesh, rank, len(processorDirs))

    config.hdf5 = True
    IOField.setMesh(mesh)
    for index, time in enumerate(times):
        fields = os.listdir(mesh.getTimeDir(time))
        for name in fields:
            phi = IOField.readFoam(name, mesh, time)
            phi.partialComplete()
            fieldWriteHDF5(phi, index, rank, len(processorDirs))

meshFile.close()
for fieldsFile in fieldsFiles:
    fieldsFile.close()
