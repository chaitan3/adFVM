#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import time
import sys

import config, parallel
from parallel import pprint
from field import IOField

from problem import primal, nSteps, writeInterval, objectiveGradient, perturb, writeResult
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

if len(sys.argv) > 2:
    firstCheckpoint = int(sys.argv[2])

# provision for restaring adjoint
if firstCheckpoint == 0:
    adjointFields = [IOField('{0}a'.format(name), np.zeros((mesh.origMesh.nInternalCells, dimensions[0]), config.precision), dimensions, mesh.calculatedBoundary) for name, dimensions in zip(primal.names, primal.dimensions)]
else:
    adjointFields = [IOField.read('{0}a'.format(name), timeSteps[nSteps - firstCheckpoint*writeInterval][0]) for name in primal.names]

# local adjoint fields
stackedAdjointFields = primal.stackFields(adjointFields, np)
pprint('STARTING ADJOINT\n')

def writeAdjointFields(writeTime):
    global adjointFields
    adjointFields = primal.unstackFields(stackedAdjointFields, IOField, names=[phi.name for phi in adjointFields])

    for phi in adjointFields:
    # TODO: fix unstacking F_CONTIGUOUS
        phi.field = np.ascontiguousarray(phi.field)
        phi.info()
        phi.write(writeTime)
    pprint()

result = 0.
for checkpoint in range(firstCheckpoint, nSteps/writeInterval):
    pprint('PRIMAL FORWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    t, dt = timeSteps[primalIndex]
    solutions = primal.run(startTime=t, dt=dt, nSteps=writeInterval, mode='forward')

    pprint('ADJOINT BACKWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    # if starting from 0, create the adjointField
    if checkpoint == 0:
        t, dt = timeSteps[-1]
        lastSolution = solutions[-1]
        stackedAdjointFields  = np.ascontiguousarray(objectiveGradient(lastSolution))
        writeAdjointFields(t)

    for step in range(0, writeInterval):
        start = time.time()

        adjointIndex = writeInterval-1 - step
        t, dt = timeSteps[primalIndex + adjointIndex]
        primal.dt.set_value(config.precision(dt))
        previousSolution = solutions[adjointIndex]
        #paddedPreviousSolution = parallel.getRemoteCells(previousSolution, mesh)
        ## adjoint time stepping
        #paddedJacobian = np.ascontiguousarray(primal.gradient(paddedPreviousSolution, stackedAdjointFields))
        #jacobian = parallel.getAdjointRemoteCells(paddedJacobian, mesh)
        jacobian = np.ascontiguousarray(primal.gradient(previousSolution, stackedAdjointFields))

        stackedAdjointFields = jacobian + np.ascontiguousarray(objectiveGradient(previousSolution))
        # compute sensitivity using adjoint solution
        perturbation = np.zeros_like(stackedAdjointFields)
        perturb(perturbation, t)
        result += np.sum(stackedAdjointFields * perturbation)

        end = time.time()
        pprint('Time for iteration: {0}'.format(end-start))
        pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

    writeAdjointFields(t)

writeResult('adjoint', result)
