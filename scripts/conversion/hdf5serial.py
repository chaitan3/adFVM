#!/usr/bin/python2
import sys
import os
import glob
import h5py
import numpy as np
import re

from adFVM import parallel, config, BCs
from adFVM.mesh import Mesh
from adFVM.field import IOField

case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]
processorDirs = glob.glob(case + 'processor*')
ranks = [int(re.search('processor([0-9]+)', proc).group(1)) for proc in processorDirs]
processorDirs = [proc for (rank, proc) in sorted(zip(ranks, processorDirs))]

def genParallel(parallelInfo, group):
    parallelInfo = np.array(parallelInfo)
    parallelEnd = np.cumsum(parallelInfo, axis=0)
    parallelStart = np.zeros_like(parallelEnd)
    parallelStart[1:] = parallelEnd[:-1]
    parallelGroup = group.create_group('parallel')
    parallelGroup.create_dataset('start', data=parallelStart)
    parallelGroup.create_dataset('end', data=parallelEnd)

def meshWriteHDF5(meshes):
    print('writing hdf5 mesh')
    ranks = len(meshes)
    parallelInfo = []
    faces = []
    points = []
    owner = []
    neighbour = []
    cells = []
    addressing = []
    for mesh in meshes:
        parallelInfo.append(np.array([mesh.faces.shape[0], mesh.points.shape[0], \
                             mesh.owner.shape[0], mesh.neighbour.shape[0],
                             mesh.cells.shape[0]
                            ]))
        parallelInfo[-1] = np.concatenate((parallelInfo[-1], [x.shape[0] for x in mesh.addressing]))
        faces.append(mesh.faces)
        points.append(mesh.points)
        owner.append(mesh.owner)
        neighbour.append(mesh.neighbour)
        cells.append(mesh.cells)
        addressing.append(mesh.addressing)

    
    genParallel(parallelInfo, meshFile)
        
    meshFile.create_dataset('faces', data=np.vstack(faces))
    meshFile.create_dataset('points', data=np.vstack(points))
    meshFile.create_dataset('owner', data=np.concatenate(owner))
    meshFile.create_dataset('neighbour', data=np.concatenate(neighbour))
    meshFile.create_dataset('cells', data=np.vstack(cells))

    names = ['pointProcAddressing', 'faceProcAddressing', 'cellProcAddressing']
    for index in range(len(addressing[0])):
        addrData = [addr[index] for addr in addressing]
        meshFile.create_dataset(names[index], data=np.concatenate(addrData))


    boundaries = []
    parallelInfo = []
    for mesh in meshes:
        boundary = []
        for patchID in mesh.boundary.keys():
            for key, value in mesh.boundary[patchID].iteritems():
                boundary.append([patchID, key, str(value)])
        boundary = np.array(boundary, dtype='S100')
        parallelInfo.append(np.array([boundary.shape[0]]))

        boundaries.append(boundary)

    boundaryGroup = meshFile.create_group('boundary')
    genParallel(parallelInfo, boundaryGroup)
    boundaryGroup.create_dataset('values', data=np.vstack(boundaries))
    boundaryGroup.create_group('fields')
    return


def fieldWriteHDF5(phis):
    name = phis[0].name
    # mesh values required outside theano
    print('writing hdf5 field {0}'.format(name))

    boundaries = []
    fields = []
    parallelInfo = []

    for phi in phis:
        boundary = []
        for patchID in phi.boundary.keys():
            patch = phi.boundary[patchID]
            for key, value in patch.iteritems():
                if not (key == 'value' and patch['type'] in BCs.valuePatches):
                    boundary.append([patchID, key, str(value)])
        boundary = np.array(boundary, dtype='S100')
        field = phi.field
        parallelInfo.append(np.array([field.shape[0], boundary.shape[0]]))

        fields.append(field)
        boundaries.append(boundary)

    fieldGroup = fieldsFile.create_group(name)


    genParallel(parallelInfo, fieldGroup)
    fieldGroup.create_dataset('field', data=np.vstack(fields))
    fieldGroup.create_dataset('boundary', data=np.vstack(boundaries))

config.hdf5 = False
meshes = []
for rank, processor in enumerate(processorDirs):
    print processor
    mesh = Mesh()
    mesh.points, mesh.faces, mesh.owner, \
            mesh.neighbour, mesh.addressing, mesh.boundary = mesh.readFoam(processor, 'constant')
    mesh.buildBeforeWrite()

    meshes.append(mesh)


meshFile = h5py.File(case + '/mesh.hdf5', 'w')
meshWriteHDF5(meshes)
meshFile.close()

for index, time in enumerate(times):
    print 'writing time', time
    if time.is_integer():
        stime = int(time)
    else:
        stime = time
    fields = {}
    fieldsFile = h5py.File(case + '/' + str(stime) + '.hdf5', 'w')

    mesh = meshes[0]
    IOField.setMesh(mesh)
    names = mesh.getFields(time)
    for name in names:
        fields = []
        for rank, processor in enumerate(processorDirs):
            print processor
            mesh = meshes[rank]
            IOField.setMesh(mesh)
            with IOField.handle(time):
                phi = IOField.readFoam(name)
                phi.partialComplete()
                fields.append(phi)

        fieldWriteHDF5(fields)
    fieldsFile.close()
