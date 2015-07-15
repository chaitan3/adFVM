#!/usr/bin/python2
from __future__ import print_function


import config, parallel
from parallel import pprint
from field import IOField
from problem import primal, nSteps, writeInterval, objectiveGradient, perturb, writeResult

import numpy as np
import time
import sys

primal.adjoint = True
mesh = primal.mesh

firstCheckpoint = 0
if parallel.rank == 0:
    timeStepFile = mesh.case + '{0}.{1}.txt'.format(nSteps, writeInterval)
    timeSteps = np.loadtxt(timeStepFile, ndmin=2)
    timeSteps = np.concatenate((timeSteps, np.array([[np.sum(timeSteps[-1]).round(9), 0]])))
else:
    timeSteps = np.zeros((nSteps + 1, 2))
parallel.mpi.Bcast(timeSteps, root=0)

# provision for restaring adjoint
adjointNames = ['{0}a'.format(name) for name in primal.names]
if firstCheckpoint == 0:
    adjointFields = [IOField(name, np.zeros((mesh.origMesh.nInternalCells, dimensions[0]), config.precision), dimensions, mesh.calculatedBoundary) for name, dimensions in zip(adjointNames, primal.dimensions)]
else:
    adjointFields = [IOField.read(name, timeSteps[nSteps - firstCheckpoint*writeInterval][0]) for name in adjointNames]

# local adjoint fields
stackedAdjointFields = primal.stackFields(adjointFields, np)
pprint('STARTING ADJOINT\n')

def writeAdjointFields(stackedAdjointFields, writeTime):
    fields = primal.unstackFields(stackedAdjointFields, IOField, names=adjointNames)
    start = time.time()
    for phi in fields:
    # TODO: fix unstacking F_CONTIGUOUS
        phi.field = np.ascontiguousarray(phi.field)
        phi.write(writeTime)
    parallel.mpi.Barrier()
    end = time.time()
    pprint('Time for writing fields: {0}'.format(end-start))
    pprint()

result = 0.


for checkpoint in range(firstCheckpoint, nSteps/writeInterval):
    pprint('PRIMAL FORWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    t, dt = timeSteps[primalIndex]
    solutions = primal.run(startTime=t, dt=dt, nSteps=writeInterval, mode='forward')

    pprint('ADJOINT BACKWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    # if starting from 0, create the adjointField
    pprint('Time marching for', ' '.join(adjointNames))

    if checkpoint == 0:
        t, dt = timeSteps[-1]
        lastSolution = solutions[-1]
        stackedAdjointFields  = np.ascontiguousarray(objectiveGradient(lastSolution)/(nSteps + 1))
        writeAdjointFields(stackedAdjointFields, t)

    for step in range(0, writeInterval):
        start = time.time()
        fields = primal.unstackFields(stackedAdjointFields, IOField, names=[phi.name for phi in adjointFields])
        for phi in fields:
            phi.info()

        adjointIndex = writeInterval-1 - step
        t, dt = timeSteps[primalIndex + adjointIndex]
        previousSolution = solutions[adjointIndex]
        #paddedPreviousSolution = parallel.getRemoteCells(previousSolution, mesh)
        ## adjoint time stepping
        #paddedJacobian = np.ascontiguousarray(primal.gradient(paddedPreviousSolution, stackedAdjointFields))
        #jacobian = parallel.getAdjointRemoteCells(paddedJacobian, mesh)
        gradients = primal.gradient(previousSolution, stackedAdjointFields, dt)
        gradient = gradients[0]
        sourceGradient = gradients[1:]

        stackedAdjointFields = np.ascontiguousarray(gradient) + np.ascontiguousarray(objectiveGradient(previousSolution)/(nSteps + 1))
        # compute sensitivity using adjoint solution
        perturbations = perturb()
        for derivative, perturbation in zip(sourceGradient, perturbations):
            result += np.sum(np.ascontiguousarray(derivative) * perturbation)

        parallel.mpi.Barrier()
        end = time.time()
        pprint('Time for adjoint iteration: {0}'.format(end-start))
        pprint('Time since beginning:', end-config.runtime)
        pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

    writeAdjointFields(stackedAdjointFields, t)

writeResult('adjoint', result)
