#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import sys

from pyRCF import Solver
from utils import ad
from field import CellField

nSteps = 1000
writeInterval = 20

primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})

def objective(fields):
    rho, rhoU, rhoE = fields
    patch = 'obstacle'
    bc = rhoE.BC
    start, end = bc[patch].startFace, bc[patch].endFace
    areas = rhoE.mesh.areas[start:end]
    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
    field = rhoE.field[start:end]
    return ad.sum(field*areas)/(nSteps + 1)

def perturb(fields):
    rho, rhoU, rhoE = fields
    patch = 'inlet'
    bc = rhoU.BC
    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
    rhoU.field[start:end][:,0] += 0.1

def writeResult(result):
    with open(primal.mesh.case + '/objective.txt', 'a') as f:
        f.write('{0}\n'.format(result))

if __name__ == "__main__":
    mesh = primal.mesh
    option = sys.argv[1]
    if option == 'orig':
        timeSteps, result = primal.run([0, 1e-2], nSteps, writeInterval, objective=objective)
        np.savetxt(primal.mesh.case + '/{0}.{1}.txt'.format(nSteps, writeInterval), timeSteps)
        writeResult(result)
    elif option == 'perturb':
        timeSteps, result = primal.run([0, 1e-2], nSteps, objective=objective, perturb=perturb)
        writeResult(result)
    elif option == 'adjoint':
        adjointFields = [CellField.read('{0}a'.format(name), mesh, 0) for name in primal.names]
        stackedAdjointFields = np.hstack([ad.value(phi.field) for phi in adjointFields])
        # write the perturbation
        result = np.sum(stackedAdjointFields*perturb())
        print(result)
    else:
        print('WTF')



