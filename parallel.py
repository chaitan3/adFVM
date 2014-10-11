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

    def exchange(self, remote, sendData, recvData, tag):
        #if isinstance(sendData, ad.adarray):
        #    sendData = ad.value(sendData)
        sendRequest = mpi.Isend([sendData, MPI.DOUBLE], dest=remote, tag=tag)
        recvRequest = mpi.Irecv([recvData, MPI.DOUBLE], source=remote, tag=tag)
        self.requests.extend([sendRequest, recvRequest])

    def wait(self):
        if nProcessors == 1:
            return
        MPI.Request.Waitall(self.requests)


