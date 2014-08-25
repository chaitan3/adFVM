#!/usr/bin/python2
from __future__ import print_function

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

a = np.random.rand(1)[0]
b = comm.allreduce(a, op=MPI.MIN)
print(b)
