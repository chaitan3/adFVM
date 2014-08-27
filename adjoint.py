#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import time
import sys

from field import CellField
from ops import derivative, forget, strip
from utils import ad

assert ad.__name__ == 'numpad'

from pyRCF import Solver
primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
def objective(fields):
    rho, rhoU, rhoE = fields
    patch = 'obstacle'
    bc = rhoE.BC
    start, end = bc[patch].startFace, bc[patch].endFace
    areas = rhoE.mesh.areas[start:end]
    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
    field = rhoE.field[start:end]
    return ad.sum(field*areas)

nSteps = 10
writeInterval = 5

if len(sys.argv) > 1:
    timeSteps = np.loadtxt(sys.argv[1])
else:
    print('PRIMAL INITIAL SWEEP: {0} Steps\n'.format(nSteps))
    timeSteps = primal.run([0, 1e-2], nSteps, writeInterval)
    np.savetxt(primal.mesh.case + '/{0}.{1}.txt'.format(nSteps, writeInterval), timeSteps)

mesh = primal.mesh
adjointFields = [CellField.zeros('{0}a'.format(name), mesh, dimensions) for name, dimensions in zip(primal.names, primal.dimensions)]
stackedAdjointFields = np.hstack([ad.value(phi.field) for phi in adjointFields])
nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in adjointFields]))))
nDimensions = zip(nDimensions[:-1], nDimensions[1:])

print('STARTING ADJOINT\n')

def writeAdjointFields(writeTime):
    for index in range(0, len(nDimensions)):
        phi = adjointFields[index]
        # range creates a copy
        phi.field = stackedAdjointFields[:, range(*nDimensions[index])]
        phi.info()
        phi.write(writeTime)
        print()

for checkpoint in range(0, nSteps/writeInterval):
    print('PRIMAL FORWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    solutions = primal.run(timeSteps[primalIndex], nSteps=writeInterval, adjoint=True)
    if checkpoint == 0:
        lastSolution = solutions[-1][-1]
        stackedAdjointFields = derivative(objective(lastSolution), lastSolution)/nSteps
        writeAdjointFields(timeSteps[-1][0])

    print('ADJOINT BACKWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))

    for step in range(0, writeInterval):
        start = time.time()

        adjointIndex = writeInterval-1 - step
        currentSolution = solutions[adjointIndex + 1][0]
        previousSolution = solutions[adjointIndex][-1]
        stackedFields = ad.hstack([phi.field for phi in currentSolution])
        jacobians = derivative(ad.sum(stackedFields*stackedAdjointFields), previousSolution)
        sensitivities = derivative(objective(previousSolution), previousSolution)/nSteps
        stackedAdjointFields = jacobians + sensitivities
        forget(currentSolution)

        end = time.time()
        print('Time for iteration: {0}\n'.format(end-start))
        print('Simulation Time and step: {0}\n'.format(timeSteps[primalIndex + adjointIndex + 1]))

    writeAdjointFields(timeSteps[primalIndex][0])
    
    
