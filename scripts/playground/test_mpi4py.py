import numpy as np
from mpi4py import MPI

mpi = MPI.COMM_WORLD

#a = np.ones((10, 3))
##b = np.zeros((20, 3))
##n = np.array([10, 10])
##m = np.cumsum(n)-n[0]
##mpi.Gatherv(a, [b, n, m, MPI.DOUBLE])
##print b, mpi.rank
#b = mpi.gather(a, root=0)
#print b
rank = mpi.rank

n = mpi.Get_size()
a = np.array([rank])
b = np.array([0])
#a = np.zeros(n)
#a[rank] = rank
#b = np.zeros(n)
mpi.Scan(a, b)
print rank, a, b
