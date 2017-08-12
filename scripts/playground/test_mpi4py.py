#!/usr/bin/python2
import numpy as np
from mpi4py import MPI
import sys

mpi = MPI.COMM_WORLD
rank = mpi.rank
n = mpi.Get_size()
hostname = MPI.Get_processor_name()

print mpi.allreduce([rank, rank+20])
print mpi.reduce([[rank], [rank+20]])
#print mpi.gather(1, root=0)
#print mpi.scatter([1], root=0)
#print mpi.bcast([1], root=0)

#print 'I\'m call', sys.argv[1], 'at', rank, 'of', n, 'processors on', hostname

#a = np.ones((10, 3))
##b = np.zeros((20, 3))
##n = np.array([10, 10])
##m = np.cumsum(n)-n[0]
##mpi.Gatherv(a, [b, n, m, MPI.DOUBLE])
##print b, mpi.rank
#b = mpi.gather(a, root=0)
#print b

#a = np.array([rank])
#b = np.array([0])
##a = np.zeros(n)
##a[rank] = rank
##b = np.zeros(n)
#mpi.Scan(a, b)
#print rank, a, b

#a = np.array([rank], dtype=np.int64)
#b = None
#if rank == 0:
#    b = np.zeros((n, 1), np.int64)
#mpi.Gather(a, b)
#print a, b
