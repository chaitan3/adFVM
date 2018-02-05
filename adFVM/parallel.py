from __future__ import print_function
import numpy as np
import subprocess
import time
import os

import multiprocessing
nProcsPerNode = multiprocessing.cpu_count()

def setNumThreads(nRanksPerNode, nProcsPerNode):
    nThreads = nProcsPerNode//nRanksPerNode
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
    #mpsRank = localRank % 2
    #localRank = 0
    #os.environ['CUDA_MPS_PIPE_DIRECTORY'] = '/tmp/nvidia-pipe-{}'.format(mpsRank)
    #os.environ['CUDA_MPS_LOG_DIRECTORY'] = '/tmp/nvidia-log-{}'.format(mpsRank)
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

    def exchange(self, remote, sendData, recvData, tag):
        sendRequest = mpi.Isend(sendData, dest=remote, tag=tag)
        recvRequest = mpi.Irecv(recvData, source=remote, tag=tag)
        self.requests.extend([sendRequest, recvRequest])

    def wait(self):
        if nProcessors == 1:
            return []
        MPI.Request.Waitall(self.requests)
        return
        
def getRemoteCells(fields, mesh, fieldTag=0):
    # mesh values required outside theano
    if nProcessors == 1:
        return fields
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
    exchanger.wait()
    return phis


