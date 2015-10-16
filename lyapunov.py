#!/usr/bin/python2
from __future__ import print_function

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
nDims = sum([dim for dim in primal.dimensions])
stackedAdjointFields = np.ones((nCells, nDims))

pprint('STARTING ADJOINT')
pprint('Number of steps:', nSteps)
pprint('Write interval:', writeInterval)
pprint()

totalCheckpoints = nSteps/writeInterval
for checkpoint in range(firstCheckpoint, totalCheckpoints):
    pprint('PRIMAL FORWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    t, dt = timeSteps[primalIndex]
    solutions = primal.run(startTime=t, dt=dt, nSteps=writeInterval, mode='forward')

    pprint('ADJOINT BACKWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
    # if starting from 0, create the adjointField
    pprint('Time marching for', ' '.join(adjointNames))

    for step in range(0, writeInterval):
        printMemUsage()
        start = time.time()

        adjointIndex = writeInterval-1 - step
        pprint('Time step', adjointIndex)
        t, dt = timeSteps[primalIndex + adjointIndex]
        previousSolution = solutions[adjointIndex]
        #paddedPreviousSolution = parallel.getRemoteCells(previousSolution, mesh)
        ## adjoint time stepping
        #paddedJacobian = np.ascontiguousarray(primal.gradient(paddedPreviousSolution, stackedAdjointFields))
        #jacobian = parallel.getAdjointRemoteCells(paddedJacobian, mesh)
        gradients = primal.gradient(previousSolution, stackedAdjointFields, dt)
        gradient = gradients[0]
        sourceGradient = gradients[1:]
        stackedAdjointFields = np.ascontiguousarray(gradient)

        parallel.mpi.Barrier()
        end = time.time()
        pprint('Time for adjoint iteration: {0}'.format(end-start))
        pprint('Time since beginning:', end-config.runtime)
        pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

    # how frequently?
    A = stackedAdjointFields.ravel()
    Q, R = np.linalg.qr(A)
    stackedAdjointFields = Q.reshape((nCells, nDims))
