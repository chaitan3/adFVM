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
import time
import sys

nLyapunov = 1

primal.adjoint = True
mesh = primal.mesh
firstCheckpoint = 0
result = 0.
if parallel.rank == 0:
    timeSteps = np.loadtxt(primal.timeStepFile, ndmin=2)
    timeSteps = np.concatenate((timeSteps, np.array([[np.sum(timeSteps[-1]).round(9), 0]])))
else:
    timeSteps = np.zeros((nSteps + 1, 2))
parallel.mpi.Bcast(timeSteps, root=0)

# local adjoint fields
nCells = mesh.origMesh.nCells
nDims = sum([dim[0] for dim in primal.dimensions])
stackedAdjointFields = np.random.rand(nLyapunov, nCells, nDims)

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

pprint('STARTING ADJOINT')
pprint('Number of steps:', nSteps)
pprint('Write interval:', writeInterval)
pprint()

totalCheckpoints = nSteps/writeInterval
exponents = []

for checkpoint in range(firstCheckpoint, totalCheckpoints):
    pprint('PRIMAL FORWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    t, dt = timeSteps[primalIndex]
    solutions = primal.run(startTime=t, dt=dt, nSteps=writeInterval, mode='forward')

    pprint('ADJOINT BACKWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
    # if starting from 0, create the adjointField
    #pprint('Time marching for', ' '.join(adjointNames))
    t0 = timeSteps[primalIndex + writeInterval][0]

    for step in range(0, writeInterval):
        printMemUsage()
        start = time.time()

        adjointIndex = writeInterval-1 - step
        pprint('Time step', adjointIndex)
        t, dt = timeSteps[primalIndex + adjointIndex]
        previousSolution = solutions[adjointIndex]
        for sim in range(0, nLyapunov):
            gradients = primal.gradient(previousSolution, stackedAdjointFields[sim], dt)
            gradient = gradients[0]
            stackedAdjointFields[sim] = np.ascontiguousarray(gradient)

        parallel.mpi.Barrier()
        end = time.time()
        pprint('Time for adjoint iteration: {0}'.format(end-start))
        pprint('Time since beginning:', end-config.runtime)
        pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

    # how frequently?
    stackedAdjointFields = parallel.gatherCells(stackedAdjointFields, mesh.origMesh, axis=1)
    if stackedAdjointFields is not None:
        totalCells = stackedAdjointFields.shape[1]
        A = stackedAdjointFields.reshape((nLyapunov, totalCells*nDims))
        Q, R = np.linalg.qr(A.T)
        S = np.diag(np.sign(np.diag(R)))
        Q, R = np.dot(Q, S), np.dot(S, R)
        r = np.log(np.diag(R))/(t0-t)
        exponents.append(r)
        stackedAdjointFields = Q.T.reshape((nLyapunov, totalCells, nDims))
    stackedAdjointFields = parallel.scatterCells(stackedAdjointFields, mesh.origMesh, axis=1)

    #for sim in range(0, nLyapunov):
    #    names = ['{0}a_{1}'.format(name, sim) for name in primal.names]
    #    writeAdjointFields(stackedAdjointFields[sim], names, t)

exponents = np.array(exponents)
pprint(exponents)
pprint(np.mean(exponents, axis=0))
