#!/usr/bin/python2
from __future__ import print_function

#from mesh import Mesh
#from field import CellField
#from interp import interpolate
#import time
#
#case = 'tests/cylinder/'
#mesh = Mesh(case)
#U = CellField.read('U', mesh, 2)
#p = CellField.read('p', mesh, 2)
#Uf = interpolate(U).field
#pf = interpolate(p).field
#
#print(mesh.sumOp.shape)
#print(pf.shape)
#print(Uf.shape)
#
#x = lambda t: mesh.sumOp * pf
#y = lambda t: mesh.sumOp * Uf
#

import numpy as np
import utils
if utils.mpi_Rank == 0:
    a = np.zeros(10)
    print(utils.mpi_Rank, a)
    r2 = utils.mpi.Irecv(a[5:], 2, 1)
    r1 = utils.mpi.Irecv(a[:5], 1, 1)
    utils.MPI.Request.Waitall([r1, r2])
    print(utils.mpi_Rank, a)
else:
    a = utils.mpi_Rank * np.arange(0., 5.)
    print(utils.mpi_Rank, a)
    utils.mpi.Send(a, 0, 1)
