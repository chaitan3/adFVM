from __future__ import print_function
from mpi4py import MPI
import numpy as np
from config import ad, T

mpi = MPI.COMM_WORLD
nProcessors = mpi.Get_size()
rank = mpi.Get_rank()
processorDirectory = '/'
if nProcessors > 1:
    processorDirectory = '/processor{0}/'.format(rank)
T.config.compiledir += '-{0}'.format(rank)
print T.config.compiledir

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


