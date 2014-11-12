#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import time
import sys

from field import IOField
from config import ad

firstCheckpoint = 0
if len(sys.argv) > 1:
    timeSteps = np.loadtxt(sys.argv[1], ndmin=2)
    timeSteps = np.concatenate((timeSteps, np.array([[np.sum(timeSteps[-1]).round(9), 0]])))
    if len(sys.argv) > 2:
        firstCheckpoint = int(sys.argv[2])
else:
    print('Primal run time step file not specified')
    exit()

from problem import primal, nSteps, writeInterval, objectiveGradient
primal.adjoint = True
mesh = primal.mesh

if firstCheckpoint == 0:
    adjointFields = [IOField('{0}a'.format(name), np.zeros((mesh.nInternalCells, dimensions[0])), dimensions, mesh.calculatedBoundary) for name, dimensions in zip(primal.names, primal.dimensions)]
else:
    adjointFields = [IOField.read('{0}a'.format(name), timeSteps[nSteps - firstCheckpoint*writeInterval][0]) for name in primal.names]

stackedAdjointFields = primal.stackFields(adjointFields, np)
print('STARTING ADJOINT\n')

def writeAdjointFields(writeTime):
    global adjointFields
    adjointFields = primal.unstackFields(stackedAdjointFields, IOField, names=[phi.name for phi in adjointFields])
    for phi in adjointFields:
        phi.info()
        phi.write(writeTime)
    print()

for checkpoint in range(firstCheckpoint, nSteps/writeInterval):
    print('PRIMAL FORWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    t, dt = timeSteps[primalIndex]
    solutions = primal.run(startTime=t, dt=dt, nSteps=writeInterval, mode='forward')

    print('ADJOINT BACKWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    if checkpoint == 0:
        lastSolution = solutions[-1]
        stackedAdjointFields  = objectiveGradient(lastSolution)
        writeAdjointFields(timeSteps[-1][0])

    for step in range(0, writeInterval):
        start = time.time()

        adjointIndex = writeInterval-1 - step
        previousSolution = solutions[adjointIndex]
        jacobians = primal.gradient(previousSolution, stackedAdjointFields)
        sensitivities = objectiveGradient(previousSolution)
        stackedAdjointFields = jacobians + sensitivities

        end = time.time()
        print('Time for iteration: {0}'.format(end-start))
        print('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

    writeAdjointFields(t)
