from __future__ import print_function
from mpi4py import MPI
import numpy as np

mpi = MPI.COMM_WORLD
nProcessors = mpi.Get_size()
rank = mpi.Get_rank()
processorDirectory = '/'
if nProcessors > 1:
    processorDirectory = '/processor{0}/'.format(rank)

def pprint(*args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)

def max(data):
    maxData = np.max(data)
    if nProcessors > 1:
        return mpi.allreduce(maxData, op=MPI.MAX)
    else:
        return maxData
def min(data):
    minData = np.min(data)
    if nProcessors > 1:
        return mpi.allreduce(minData, op=MPI.MIN)
    else:
        return minData
    
class Exchanger(object):
    def __init__(self):
        self.requests = []
        self.statuses = []

    def exchange(self, remote, sendData, recvData, tag):
        sendRequest = mpi.Isend(sendData, dest=remote, tag=tag)
        recvRequest = mpi.Irecv(recvData, source=remote, tag=tag)
        sendStatus = MPI.Status()
        recvStatus = MPI.Status()
        self.requests.extend([sendRequest, recvRequest])
        self.statuses.extend([sendStatus, recvStatus])

    def wait(self):
        if nProcessors == 1:
            return []
        MPI.Request.Waitall(self.requests, self.statuses)
        return self.statuses

def getRemoteCells(stackedFields, mesh):
    #logger.info('fetching remote cells')
    if nProcessors == 1:
        return stackedFields

    origMesh = mesh
    mesh = mesh.paddedMesh 
    precision = stackedFields.dtype

    paddedStackedFields = np.zeros((mesh.nCells, ) + stackedFields.shape[1:], precision)
    paddedStackedFields[:origMesh.nInternalCells] = stackedFields[:origMesh.nInternalCells]
    nLocalBoundaryFaces = origMesh.nLocalCells - origMesh.nInternalCells
    paddedStackedFields[mesh.nInternalCells:mesh.nInternalCells + nLocalBoundaryFaces] = stackedFields[origMesh.nInternalCells:origMesh.nLocalCells]

    exchanger = Exchanger()
    internalCursor = origMesh.nInternalCells
    boundaryCursor = origMesh.nCells
    for patchID in origMesh.remotePatches:
        nInternalCells = mesh.remoteCells['internal'][patchID]
        nBoundaryCells = mesh.remoteCells['boundary'][patchID]
        local, remote, tag = origMesh.getProcessorPatchInfo(patchID)
        exchanger.exchange(remote, stackedFields[mesh.localRemoteCells['internal'][patchID]], paddedStackedFields[internalCursor:internalCursor+nInternalCells], tag)
        tag += len(origMesh.origPatches) + 1
        exchanger.exchange(remote, stackedFields[mesh.localRemoteCells['boundary'][patchID]], paddedStackedFields[boundaryCursor:boundaryCursor+nBoundaryCells], tag)
        internalCursor += nInternalCells
        boundaryCursor += nBoundaryCells
    exchanger.wait()

    # second round of transferring: does not matter which processor
    # the second layer belongs to, in previous transfer the correct values have been put in 
    # the extra remote ghost cells, transfer that portion
    exchanger = Exchanger()
    boundaryCursor = origMesh.nCells
    for patchID in origMesh.remotePatches:
        nBoundaryCells = mesh.remoteCells['boundary'][patchID]
        boundaryCursor += nBoundaryCells
        nExtraRemoteBoundaryCells = mesh.remoteCells['extra'][patchID]
        nLocalBoundaryCells = origMesh.nLocalCells - origMesh.nInternalCells
        local, remote, tag = origMesh.getProcessorPatchInfo(patchID)
        # does it work if sendData/recvData is empty
        #print patchID, nExtraRemoteBoundaryCells, len(mesh.localRemoteCells['extra'][patchID])
        exchanger.exchange(remote, paddedStackedFields[-nLocalBoundaryCells + mesh.localRemoteCells['extra'][patchID]], paddedStackedFields[boundaryCursor-nExtraRemoteBoundaryCells:boundaryCursor], tag)
    exchanger.wait()

    return paddedStackedFields

def getAdjointRemoteCells(paddedJacobian, mesh):
    #logger.info('fetching adjoint remote cells')
    if nProcessors == 1:
        return paddedJacobian

    origMesh = mesh
    mesh = mesh.paddedMesh 
    precision = paddedJacobian.dtype
    dimensions = paddedJacobian.shape[1:]

    jacobian = np.zeros(((origMesh.nCells,) + dimensions), precision)
    jacobian[:origMesh.nInternalCells] = paddedJacobian[:origMesh.nInternalCells]
    nLocalBoundaryFaces = origMesh.nLocalCells - origMesh.nInternalCells
    jacobian[origMesh.nInternalCells:origMesh.nLocalCells] = paddedJacobian[mesh.nInternalCells:mesh.nInternalCells+nLocalBoundaryFaces]

    exchanger = Exchanger()
    internalCursor = origMesh.nInternalCells
    boundaryCursor = origMesh.nCells
    adjointRemoteCells = {'internal':{}, 'boundary':{}, 'extra':{}}
    for patchID in origMesh.remotePatches:
        nInternalCells = mesh.remoteCells['internal'][patchID]
        nBoundaryCells = mesh.remoteCells['boundary'][patchID]
        local, remote, tag = origMesh.getProcessorPatchInfo(patchID)
        
        size = (len(mesh.localRemoteCells['internal'][patchID]), ) + dimensions
        adjointRemoteCells['internal'][patchID] = np.zeros(size, precision)
        exchanger.exchange(remote, paddedJacobian[internalCursor:internalCursor+nInternalCells], adjointRemoteCells['internal'][patchID], tag)
        internalCursor += nInternalCells
        tag += len(origMesh.origPatches) + 1

        size = (len(mesh.localRemoteCells['boundary'][patchID]), ) + dimensions
        adjointRemoteCells['boundary'][patchID] = np.zeros(size, precision)
        exchanger.exchange(remote, paddedJacobian[boundaryCursor:boundaryCursor+nBoundaryCells], adjointRemoteCells['boundary'][patchID], tag)
        boundaryCursor += nBoundaryCells
        tag += len(origMesh.origPatches) + 1

    exchanger.wait()
    for patchID in origMesh.remotePatches:
        jacobian[mesh.localRemoteCells['internal'][patchID]] += adjointRemoteCells['internal'][patchID]
        jacobian[mesh.localRemoteCells['boundary'][patchID]] += adjointRemoteCells['boundary'][patchID]

    # code for second layer: transfer to remote jacobians again and add up
    exchanger = Exchanger()
    internalCursor = origMesh.nLocalCells
    for patchID in origMesh.remotePatches:
        nInternalCells = len(mesh.localRemoteCells['internal'][patchID])
        local, remote, tag = origMesh.getProcessorPatchInfo(patchID)
        
        size = (nInternalCells, ) + dimensions
        adjointRemoteCells['extra'][patchID] = np.zeros(size, precision)
        exchanger.exchange(remote, jacobian[internalCursor:internalCursor+nInternalCells], adjointRemoteCells['extra'][patchID], tag)
        internalCursor += nInternalCells

    exchanger.wait()
    for patchID in origMesh.remotePatches:
        jacobian[mesh.localRemoteCells['internal'][patchID]] += adjointRemoteCells['extra'][patchID]

    # make processor cells zero again
    jacobian[origMesh.nLocalCells:] = 0.

    return jacobian
