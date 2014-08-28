#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import sys

from pyRCF import Solver
from utils import ad
from field import CellField

nSteps = 1000
writeInterval = 50

primal = Solver('tests/convection/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})

def objective(fields):
    rho, rhoU, rhoE = fields
    mesh = rho.mesh
    mid = np.array([0.75, 0.5, 0.5])
    indices = range(0, mesh.nInternalCells)
    G = np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1)*mesh.volumes[indices]
    return ad.sum(rho.field[indices]*G)

def perturb(fields):
    rho, rhoU, rhoE = fields
    mesh = rho.mesh
    mid = np.array([0.25, 0.5, 0.5])
    indices = range(0, mesh.nInternalCells)
    G = 1e-4*np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1)
    rho.field[indices] += G

#primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})

#def objective(fields):
#    rho, rhoU, rhoE = fields
#    patch = 'obstacle'
#    bc = rhoE.BC
#    start, end = bc[patch].startFace, bc[patch].endFace
#    areas = rhoE.mesh.areas[start:end]
#    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
#    field = rhoE.field[start:end]
#    return ad.sum(field*areas)/(nSteps + 1)


#def perturb(fields):
#    rho, rhoU, rhoE = fields
#    patch = 'inlet'
#    bc = rhoU.BC
#    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
#    rhoU.field[start:end][:,0] += 0.1

if __name__ == "__main__":
    mesh = primal.mesh
    option = sys.argv[1]
    if option == 'orig':
        timeSteps, result = primal.run([0, 1e-2], nSteps, writeInterval, objective=objective)
        np.savetxt(primal.mesh.case + '/{0}.{1}.txt'.format(nSteps, writeInterval), timeSteps)
    elif option == 'perturb':
        timeSteps, result = primal.run([0, 1e-2], nSteps, objective=objective, perturb=perturb)
    elif option == 'adjoint':
        adjointFields = [CellField.read('{0}a'.format(name), mesh, 0) for name in primal.names]
        stackedAdjointFields = np.hstack([ad.value(phi.field) for phi in adjointFields])
        fields = [CellField.zeros(name, mesh, dimension) for name, dimension in zip(primal.names, primal.dimensions)]
        perturb(fields)
        stackedFields = np.hstack([ad.value(phi.field) for phi in fields])
        result = np.sum(stackedAdjointFields*stackedFields)
    else:
        print('WTF')
        exit()

    with open(primal.mesh.case + '/objective.txt', 'a') as f:
        f.write('{0} {1}\n'.format(option, result))

