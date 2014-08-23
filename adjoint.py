#!/usr/bin/python2
from __future__ import print_function
from field import CellField
from ops import derivative, forget, strip
import numpad as ad
import numpy as np
import time

from pyRCF import Solver
def objective(fields):
    rho, rhoU, rhoE = fields
    patch = 'obstacle'
    bc = rhoE.BC
    start, end = bc[patch].startFace, bc[patch].endFace
    areas = rhoE.mesh.areas[start:end]
    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
    field = rhoE.field[start:end]
    return ad.sum(field*areas)
primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 2.5e-5, 'Pr': 0.7, 'CFL': 0.2})

nSteps = 10
writeInterval = 2

print('PRIMAL INITIAL SWEEP: {0} Steps\n'.format(nSteps))
mesh = primal.mesh
timeSteps = primal.run([0, 1e-2], nSteps, writeInterval)

adjointFields = [CellField.zeros('{0}a'.format(name), mesh, dimensions) for name, dimensions in zip(primal.names, primal.dimensions)]
nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in adjointFields]))))
nDimensions = zip(nDimensions[:-1], nDimensions[1:])
adjoint = np.ravel(np.hstack([ad.value(phi.field) for phi in adjointFields]))

print('STARTING ADJOINT\n')

for checkpoint in range(0, nSteps/writeInterval):
    print('PRIMAL FORWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    solutions = primal.run(timeSteps[primalIndex], nSteps=writeInterval, adjoint=True)
    print('ADJOINT BACKWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))

    for step in range(0, writeInterval):
        print('Adjoint Step {0}'.format(step))
        start = time.time()

        adjointIndex = writeInterval-1 - step
        stackedFields = ad.ravel(ad.hstack([phi.field for phi in solutions[adjointIndex + 1]]))
        jacobian = derivative(ad.sum(stackedFields*adjoint), solutions[adjointIndex])
        strippedSolution = strip(solutions[adjointIndex])
        sensitivity = derivative(objective(strippedSolution), strippedSolution)/nSteps
        print(jacobian.min(), jacobian.max())
        print(sensitivity.min(), sensitivity.max())
        adjoint = jacobian + sensitivity

        end = time.time()
        print('Time for iteration: {0}\n'.format(end-start))
    
    adjointMat = adjoint.reshape((mesh.nCells, 5))
    for index in range(0, len(nDimensions)):
        phi = adjointFields[index]
        phi.field = adjointMat[:, range(*nDimensions[index])]
        phi.write(timeSteps[primalIndex][0])
