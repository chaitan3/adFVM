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

    paddedStackedFields = np.zeros((mesh.nCells, ) + stackedFields.shape[1:], config.precision)
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
    # the second layer belongs to
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

    paddedMesh = mesh.paddedMesh

    jacobian = np.zeros(((mesh.nCells,) + paddedJacobian.shapep[1:]), config.precision)
    jacobian[:mesh.nInternalCells] = paddedJacobian[:mesh.nInternalCells]
    nLocalBoundaryFaces = mesh.nLocalCells - mesh.nInternalCells
    jacobian[mesh.nInternalCells:mesh.nLocalCells] = paddedJacobian[paddedMesh.nInternalCells:paddedMesh.nInternalCells+nLocalBoundaryFaces]

    exchanger = Exchanger()
    internalCursor = origMesh.nInternalCells
    boundaryCursor = origMesh.nCells
    for patchID in origMesh.remotePatches:
        nInternalCells = mesh.remoteCells['internal'][patchID]
        nBoundaryCells = mesh.remoteCells['boundary'][patchID]
        local, remote, tag = origMesh.getProcessorPatchInfo(patchID)
        exchanger.exchange(remote, paddedJacobian[mesh.localRemoteCells['internal'][patchID]], jacobian[internalCursor:internalCursor+nInternalCells], tag)
        tag += len(origMesh.origPatches) + 1
        exchanger.exchange(remote, paddedJacobian[mesh.localRemoteCells['boundary'][patchID]], jacobian[boundaryCursor:boundaryCursor+nBoundaryCells], tag)
        internalCursor += nInternalCells
        boundaryCursor += nBoundaryCells
    exchanger.wait()

    return jacobian

