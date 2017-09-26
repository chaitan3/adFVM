from __future__ import print_function
import numpy as np
import subprocess
import time
import os

import multiprocessing
nProcsPerNode = multiprocessing.cpu_count()

def setNumThreads(nRanksPerNode, nProcsPerNode):
    nThreads = nProcsPerNode/nRanksPerNode
    os.environ['OMP_NUM_THREADS'] = str(max(1, nThreads))

def getLocalRank(mpi, name, rank):
    nameRanks = mpi.gather((name, rank), root=0)
    names = {}
    if rank == 0:
        for n, r in nameRanks:
            if n not in names:
                names[n] = []
            names[n].append(r)
        for n in names:
            names[n].sort()
    names = mpi.bcast(names, root=0)
    return names[name].index(rank), len(names[name])

try:
    from mpi4py import MPI
    mpi = MPI.COMM_WORLD
    nProcessors = mpi.Get_size()
    name = MPI.Get_processor_name()
    rank = mpi.Get_rank()
    localRank, nRanksPerNode = getLocalRank(mpi, name, rank)
except:
    print('mpi4py NOT LOADED: YOU SHOULD BE RUNNING ON A SINGLE CORE')
    class Container(object):
        pass
    mpi = Container()
    nProcessors = 1
    name = ''
    rank = 0
    localRank, nRanksPerNode = 0, 1
    mpi.bcast = lambda x, root: x
    mpi.Bcast = lambda x, root: None
    mpi.Barrier = lambda : None
    mpi.scatter = lambda x, root: x[0]
    mpi.gather = lambda x, root: [x]

setNumThreads(nRanksPerNode, nProcsPerNode)

processorDirectory = '/'
if nProcessors > 1:
    processorDirectory = '/processor{0}/'.format(rank)
temp = '/tmp'

def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)
pprint('Running on {0} processors'.format(nProcessors))

def copyToTemp(home, coresPerNode):
    start = time.time()
    if rank % coresPerNode == 0:
        dest = temp + '/.theano'
        subprocess.call(['rm', '-rf', dest])
        subprocess.call(['cp', '-r', home + '/.theano', dest])
    mpi.Barrier()
    end = time.time()
    pprint('Time to copy to {0}: '.format(temp), end-start)
    pprint()
    return temp

def reduction(data, op, allreduce):
    if not isinstance(data, list):
        data = [data]
        singleton = True
    else:
        singleton = False
    np_op, mpi_op = op
    redData = []
    n = len(data)
    for d in data:
        redData.append(np_op(d))
    if nProcessors > 1:
        redData = np.array(redData)
        ret = np.zeros(n, redData.dtype)
        if allreduce:
            mpi.Allreduce(redData, ret, op=mpi_op)
        else:
            mpi.Reduce(redData, ret, op=mpi_op, root=0)
        ret = ret.tolist()
    else:
        ret = redData                           
    if singleton:
        return ret[0]
    else:
        return ret

def max(data, allreduce=True):
    return reduction(data, (np.max, MPI.MAX), allreduce)

def min(data, allreduce=True):
    return reduction(data, (np.min, MPI.MIN), allreduce)

def sum(data, allreduce=True):
    return reduction(data, (np.sum, MPI.SUM), allreduce)

def argmin(data):
    minData, index = np.min(data), np.argmin(data)
    if nProcessors > 1:
        proc = mpi.allreduce([minData, rank], op=MPI.MINLOC) 
        if rank == proc[1]:
            return [index]
        else:
            return []
    return [index]

class Exchanger(object):
    def __init__(self):
        self.requests = []
        #self.statuses = []

    def exchange(self, remote, sendData, recvData, tag):
        sendRequest = mpi.Isend(sendData, dest=remote, tag=tag)
        recvRequest = mpi.Irecv(recvData, source=remote, tag=tag)
        self.requests.extend([sendRequest, recvRequest])
        #sendStatus = MPI.Status()
        #recvStatus = MPI.Status()
        #self.statuses.extend([sendStatus, recvStatus])

    def wait(self):
        if nProcessors == 1:
            return []
        MPI.Request.Waitall(self.requests)
        return
        #try:
        #    MPI.Request.Waitall(self.requests, self.statuses)
        #    print(rank, 'done')
        #except:
        #    print(rank, 'send', MPI.Get_error_string(self.statuses[0].Get_error()))
        #    print(rank, 'recv', MPI.Get_error_string(self.statuses[1].Get_error()))
        #    exit(0)
        #return self.statuses

#mtime = 0.
#wtime = 0.
def getRemoteCells(fields, mesh, fieldTag=0):
    # mesh values required outside theano
    #logger.info('fetching remote cells')
    if nProcessors == 1:
        return fields
    #global mtime, wtime
    #if meshC.reset:
    #    meshC.reset = False
    #    #print(rank, mtime, wtime)
    #    mtime = 0.
    #    wtime = 0.

    #start = time.time()
    #mpi.Barrier()
    #start2 = time.time()
    #wtime += start2-start
    exchanger = Exchanger()
    phis = []
    for index, field in enumerate(fields):
        assert field.shape[0] <= mesh.nCells
        assert field.shape[0] >= mesh.nLocalCells
        phi = np.concatenate((field[:mesh.nLocalCells].copy(), np.zeros((mesh.nCells-mesh.nLocalCells,) + field.shape[1:], field.dtype)))
        phis.append(phi)
        assert field.flags['C_CONTIGUOUS']
        for patchID in mesh.remotePatches:
            local, remote, tag = mesh.getProcessorPatchInfo(patchID)
            tag += 1000*index
            startFace, endFace, cellStartFace, cellEndFace, _ = mesh.getPatchFaceCellRange(patchID)
            exchanger.exchange(remote, phi[mesh.owner[startFace:endFace]], phi[cellStartFace:cellEndFace], fieldTag + tag)
    #print(rank, fieldTag)
    exchanger.wait()
        #mtime += time.time()-start2
    return phis

def getRemoteFaces(field1, field2, procStartFace, mesh):
    # mesh values required outside theano
    #logger.info('fetching remote cells')
    if nProcessors == 1:
        return field2
    exchanger = Exchanger()
    startFace = procStartFace.copy()
    #print(field2[procStartFace:])
    for patchID in mesh.remotePatches:
        local, remote, tag = mesh.getProcessorPatchInfo(patchID)
        _, _, nFaces = mesh.getPatchFaceRange(patchID)
        endFace = startFace + nFaces
        exchanger.exchange(remote, field1[startFace:endFace], field2[startFace:endFace], tag)
        startFace += nFaces
    exchanger.wait()
    #mtime += time.time()-start2
    return field2

def getAdjointRemoteFaces(field, procStartFace, mesh):
    if nProcessors == 1:
        return np.zeros_like(field), field
    jacobian1 = np.zeros_like(field)
    jacobian2 = np.zeros_like(field)
    jacobian2[:procStartFace] = field[:procStartFace]
    precision = field.dtype
    dimensions = field.shape[1:]
    adjointRemoteFaces = {}
    exchanger = Exchanger()
    startFace = procStartFace.copy()
    for patchID in mesh.remotePatches:
        local, remote, tag = mesh.getProcessorPatchInfo(patchID)
        _, _, nFaces = mesh.getPatchFaceRange(patchID)
        endFace = startFace + nFaces
        adjointRemoteFaces[patchID] = np.zeros((nFaces,) + dimensions, precision)
        exchanger.exchange(remote, field[startFace:endFace], adjointRemoteFaces[patchID], tag)
        startFace += nFaces
    exchanger.wait()
    startFace = procStartFace.copy()
    for patchID in mesh.remotePatches:
        _, _, nFaces = mesh.getPatchFaceRange(patchID)
        endFace = startFace + nFaces
        jacobian1[startFace:endFace] += adjointRemoteFaces[patchID]
        startFace += nFaces
    return jacobian1, jacobian2

def getAdjointRemoteCells(fields, mesh, fieldTag=0):
    if nProcessors == 1:
        return fields
    #print(type(field), field.shape, mesh.nLocalCells)
    #print(rank, fieldTag, len(fields))

    phis = []
    adjointRemoteCells = []
    exchanger = Exchanger()
    for index, field in enumerate(fields):
        phi = np.zeros_like(field)
        phi[:mesh.nLocalCells] = field[:mesh.nLocalCells]
        phis.append(phi)
        precision = field.dtype
        dimensions = field.shape[1:]
        adjointRemoteCells.append({})
        for patchID in mesh.remotePatches:
            local, remote, tag = mesh.getProcessorPatchInfo(patchID)
            tag += 1000*index
            startFace, endFace, cellStartFace, cellEndFace, nFaces = mesh.getPatchFaceCellRange(patchID)
            adjointRemoteCells[-1][patchID] = np.zeros((nFaces,) + dimensions, precision)
            exchanger.exchange(remote, field[mesh.neighbour[startFace:endFace]], adjointRemoteCells[-1][patchID], fieldTag + tag)
    exchanger.wait()
    for phi, field, adj in zip(phis, fields, adjointRemoteCells):
        for patchID in mesh.remotePatches:
            startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
            np.add.at(phi, mesh.owner[startFace:endFace], adj[patchID])
    return phis

def gatherCells(field, mesh, axis=0):
    if nProcessors == 1:
        return field
    #nCells = np.array(mpi.gather(mesh.nCells))
    #nCellsPos = np.cumsum(nCells)-nCells[0]
    #if rank == 0:
    #    totalField = np.zeros((np.sum(nCells),) + field.shape[1:])
    #else:
    #    totalField = None
    #mpi.Gatherv(field, [totalField, nCells, nCellsPos])
    #return totalField
    totalField = mpi.gather(field)
    if totalField is not None:
        totalField = np.concatenate(totalField, axis=axis)
    return totalField

def scatterCells(totalField, mesh, axis=0):
    if nProcessors == 1:
        return totalField
    nCells = np.array(mpi.gather(mesh.nCells))
    if totalField is not None:
        nCellsPos = np.cumsum(nCells)[:-1]
        #field = np.zeros((mesh.nCells,) + totalField.shape[1:])
        #mpi.Scatterv([totalField, nCells, nCellsPos], field)
        #return field
        totalField = np.split(totalField, nCellsPos, axis=axis)
    field = mpi.scatter(totalField)
    return field

#def getRemoteCells(stackedFields, mesh):
#    # mesh values required outside theano
#    #logger.info('fetching remote cells')
#    if nProcessors == 1:
#        return stackedFields
#
#    meshC = mesh
#    origMesh = mesh.origMesh
#    meshP = mesh.paddedMesh 
#    mesh = meshP.origMesh
#    precision = stackedFields.dtype
#
#    paddedStackedFields = np.zeros((mesh.nCells, ) + stackedFields.shape[1:], precision)
#    paddedStackedFields[:origMesh.nInternalCells] = stackedFields[:origMesh.nInternalCells]
#    nLocalBoundaryFaces = origMesh.nLocalCells - origMesh.nInternalCells
#    paddedStackedFields[mesh.nInternalCells:mesh.nInternalCells + nLocalBoundaryFaces] = stackedFields[origMesh.nInternalCells:origMesh.nLocalCells]
#
#    exchanger = Exchanger()
#    internalCursor = origMesh.nInternalCells
#    boundaryCursor = origMesh.nCells
#    for patchID in meshC.remotePatches:
#        nInternalCells = len(meshP.remoteCells['internal'][patchID])
#        nBoundaryCells = len(meshP.remoteCells['boundary'][patchID])
#        local, remote, tag = meshC.getProcessorPatchInfo(patchID)
#        exchanger.exchange(remote, stackedFields[meshP.localRemoteCells['internal'][patchID]], paddedStackedFields[internalCursor:internalCursor+nInternalCells], tag)
#        tag += meshC.nLocalPatches + 1
#        exchanger.exchange(remote, stackedFields[meshP.localRemoteCells['boundary'][patchID]], paddedStackedFields[boundaryCursor:boundaryCursor+nBoundaryCells], tag)
#        internalCursor += nInternalCells
#        boundaryCursor += nBoundaryCells
#    exchanger.wait()
#
#    # second round of transferring: does not matter which processor
#    # the second layer belongs to, in previous transfer the correct values have been put in 
#    # the extra remote ghost cells, transfer that portion
#    exchanger = Exchanger()
#    boundaryCursor = origMesh.nCells
#    for patchID in meshC.remotePatches:
#        nBoundaryCells = len(meshP.remoteCells['boundary'][patchID])
#        boundaryCursor += nBoundaryCells
#        nExtraRemoteBoundaryCells = len(meshP.remoteCells['extra'][patchID])
#        nLocalBoundaryCells = origMesh.nLocalCells - origMesh.nInternalCells
#        local, remote, tag = meshC.getProcessorPatchInfo(patchID)
#        # does it work if sendData/recvData is empty
#        exchanger.exchange(remote, paddedStackedFields[-nLocalBoundaryCells + meshP.localRemoteCells['extra'][patchID]], paddedStackedFields[boundaryCursor-nExtraRemoteBoundaryCells:boundaryCursor], tag)
#    exchanger.wait()
#
#    return paddedStackedFields
#
#def getAdjointRemoteCells(paddedJacobian, mesh):
#    # mesh values required outside theano
#    #logger.info('fetching adjoint remote cells')
#    if nProcessors == 1:
#        return paddedJacobian
#
#    meshC = mesh
#    meshP = mesh.paddedMesh 
#    origMesh = mesh.origMesh
#    mesh = meshP.origMesh
#    precision = paddedJacobian.dtype
#    dimensions = paddedJacobian.shape[1:]
#
#    jacobian = np.zeros(((origMesh.nCells,) + dimensions), precision)
#    jacobian[:origMesh.nInternalCells] = paddedJacobian[:origMesh.nInternalCells]
#    nLocalBoundaryFaces = origMesh.nLocalCells - origMesh.nInternalCells
#    jacobian[origMesh.nInternalCells:origMesh.nLocalCells] = paddedJacobian[mesh.nInternalCells:mesh.nInternalCells+nLocalBoundaryFaces]
#
#    exchanger = Exchanger()
#    internalCursor = origMesh.nInternalCells
#    boundaryCursor = origMesh.nCells
#    adjointRemoteCells = {'internal':{}, 'boundary':{}, 'extra':{}}
#    for patchID in meshC.remotePatches:
#        nInternalCells = len(meshP.remoteCells['internal'][patchID])
#        nBoundaryCells = len(meshP.remoteCells['boundary'][patchID])
#        local, remote, tag = meshC.getProcessorPatchInfo(patchID)
#        
#        size = (len(meshP.localRemoteCells['internal'][patchID]), ) + dimensions
#        adjointRemoteCells['internal'][patchID] = np.zeros(size, precision)
#        exchanger.exchange(remote, paddedJacobian[internalCursor:internalCursor+nInternalCells], adjointRemoteCells['internal'][patchID], tag)
#        internalCursor += nInternalCells
#        tag += meshC.nLocalPatches + 1
#
#        size = (len(meshP.localRemoteCells['boundary'][patchID]), ) + dimensions
#        adjointRemoteCells['boundary'][patchID] = np.zeros(size, precision)
#        exchanger.exchange(remote, paddedJacobian[boundaryCursor:boundaryCursor+nBoundaryCells], adjointRemoteCells['boundary'][patchID], tag)
#        boundaryCursor += nBoundaryCells
#        tag += meshC.nLocalPatches + 1
#
#    exchanger.wait()
#    for patchID in meshC.remotePatches:
#        add_at(jacobian, meshP.localRemoteCells['internal'][patchID], adjointRemoteCells['internal'][patchID])
#        add_at(jacobian, meshP.localRemoteCells['boundary'][patchID], adjointRemoteCells['boundary'][patchID])
#
#    # code for second layer: transfer to remote jacobians again and add up
#    exchanger = Exchanger()
#    internalCursor = origMesh.nLocalCells
#    for patchID in meshC.remotePatches:
#        nInternalCells = len(meshP.localRemoteCells['internal'][patchID])
#        local, remote, tag = meshC.getProcessorPatchInfo(patchID)
#        
#        size = (nInternalCells, ) + dimensions
#        adjointRemoteCells['extra'][patchID] = np.zeros(size, precision)
#        exchanger.exchange(remote, jacobian[internalCursor:internalCursor+nInternalCells], adjointRemoteCells['extra'][patchID], tag)
#        internalCursor += nInternalCells
#
#    exchanger.wait()
#    for patchID in meshC.remotePatches:
#        add_at(jacobian, meshP.localRemoteCells['internal'][patchID], adjointRemoteCells['extra'][patchID])
#
#    # make processor cells zero again
#    jacobian[origMesh.nLocalCells:] = 0.
#
#    return jacobian
#
#
