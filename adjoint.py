#!/usr/bin/python2
from __future__ import print_function
from field import CellField
import numpad as ad
import numpy as np
import time

from pyRCF import Solver
def objective(solver):
    patch = 'obstacle' 
    U, T, p = solver.primitive(solver.rho, solver.rhoU, solver.rhoE) 
    bc = solver.p.BC
    start, end = bc[patch].startFace, bc[patch].endFace
    areas = solver.mesh.areas[start:end]
    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
    field = p.field[start:end]
    return ad.sum(field*areas)
primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 2.5e-5, 'Pr': 0.7, 'CFL': 0.2})

nSteps = 2
writeInterval = 1

print('PRIMAL INITIAL SWEEP: {0} Steps\n'.format(nSteps))
mesh = primal.mesh
timeSteps = primal.run([0, 1e-4], nSteps, writeInterval)[0]

adjointFields = [CellField.zeros('{0}a'.format(phi.name), phi.mesh, phi.dimensions) for phi in primal.fields]
nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in adjointFields]))))
nDimensions = zip(nDimensions[:-1], nDimensions[1:])
adjoint = np.hstack([ad.value(phi.field) for phi in adjointFields]).ravel()

print('STARTING ADJOINT\n')

for checkpoint in range(0, nSteps/writeInterval):
    print('PRIMAL FORWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    jacobians, sensitivities = primal.run(timeSteps[primalIndex], nSteps=writeInterval, adjoint=True, objective=objective)[1:]
    print('ADJOINT BACKWARD RUN {0}: {1} Steps\n'.format(checkpoint, writeInterval))
    for step in range(0, writeInterval):
        print('Adjoint Step {0}'.format(step))
        start = time.time()
        adjointIndex = writeInterval-1 - step
        print(jacobians[adjointIndex].shape)
        print(adjoint.shape)
        print(sensitivities[adjointIndex].shape)
        adjoint = jacobians[adjointIndex].transpose().dot(adjoint) + sensitivities[adjointIndex]/nSteps
        end = time.time()
        print('Time for iteration: {0}\n'.format(end-start))
    

    adjointMat = adjoint.reshape((mesh.nCells, 5))
    for index in range(0, len(nDimensions)):
        phi = adjointFields[index]
        phi.field = ad.array(adjointMat[:, range(*nDimensions[index])])
        phi.write(timeSteps[primalIndex][0])


