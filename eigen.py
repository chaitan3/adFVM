#!/usr/bin/python2
#from __future__ import print_function

import config, parallel
from config import ad
from parallel import pprint
from field import IOField, Field
#from op import laplacian
from matop import laplacian, ddt, BCs
from interp import central
from problem import primal, nSteps, writeInterval
from compat import printMemUsage

import numpy as np
from scipy.sparse import linalg
import time
import sys

nEigen = 1
primal.adjoint = True
mesh = primal.mesh
#if parallel.rank == 0:
#    timeSteps = np.loadtxt(primal.timeStepFile, ndmin=2)
#    timeSteps = np.concatenate((timeSteps, np.array([[np.sum(timeSteps[-1]).round(9), 0]])))
#else:
#    timeSteps = np.zeros((nSteps + 1, 2))
#parallel.mpi.Bcast(timeSteps, root=0)
#t, dt = timeSteps[nSteps-writeInterval]
t, dt = 1.120001033,2.279e-3
fields = primal.initFields(t)
stackedFields = primal.stackFields(fields, np)
primal.compile()

# local adjoint fields
nCells = mesh.origMesh.nCells
nDims = sum([dim[0] for dim in primal.dimensions])

def writeAdjointFields(stackedAdjointFields, names, writeTime):
    # TODO: fix unstacking F_CONTIGUOUS
    start = time.time()
    fields = primal.unstackFields(stackedAdjointFields, IOField, names=names, boundary=mesh.calculatedBoundary)
    for phi in fields:
        phi.field = np.ascontiguousarray(phi.field)
        phi.write(writeTime)
    parallel.mpi.Barrier()
    end = time.time()
    pprint('Time for writing fields: {0}'.format(end-start))
    pprint()


def adjoint(stackedAdjointFields):
    stackedAdjointFields = parallel.scatterCells(stackedAdjointFields, mesh.origMesh)
    gradient = primal.gradient(stackedFields, stackedAdjointFields, dt)[0]
    gradient = parallel.gatherCells(gradient, mesh.origMesh)
    return gradient

totalCells = parallel.sum(nCells)
N = totalCells*nDims
def operator(vector):
    if not hasattr(operator, 'count'):
        operator.count = 0
    operator.count += 1
    pprint(operator.count, 'operation')
    stackedAdjointFields = vector.reshape((totalCells, nDims))
    parallel.mpi.bcast(True)
    gradient = adjoint(stackedAdjointFields)
    return gradient.reshape(N)

vectors = [None for sim in range(0, nEigen)]
if parallel.rank == 0:
    A = linalg.LinearOperator((N,N), matvec=operator)
    values, vectors = linalg.eigs(A, nEigen, tol=1e-6)
    vectors = vectors.reshape((totalCells, nDims, nEigen))
    vectors = [vectors[:,:,i] for i in range(0, nEigen)]
    parallel.mpi.bcast(False)
else:
    run = parallel.mpi.bcast(None)
    while run:
        adjoint(None)
        run = parallel.mpi.bcast(None)

for sim in range(0, nEigen):
    stackedAdjointFields = parallel.scatterCells(vectors[sim], mesh.origMesh)
    names = ['{0}a_{1}'.format(name, sim) for name in primal.names]
    writeAdjointFields(stackedAdjointFields, names, t)
