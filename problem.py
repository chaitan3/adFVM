#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import sys

from pyRCF import Solver
from utils import ad
from field import CellField
from ops import strip, explicit, derivative

nSteps = 20000
writeInterval = 100

#primal = Solver('tests/convection/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
#
#def objective(fields):
#    rho, rhoU, rhoE = fields
#    mesh = rho.mesh
#    mid = np.array([0.75, 0.5, 0.5])
#    indices = range(0, mesh.nInternalCells)
#    G = np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1)*mesh.volumes[indices]
#    return ad.sum(rho.field[indices]*G)/(nSteps + 1)
#
#def perturb(fields, eps=1E-2):
#    rho, rhoU, rhoE = fields
#    mesh = rho.mesh
#    mid = np.array([0.5, 0.5, 0.5])
#    indices = range(0, mesh.nInternalCells)
#    G = eps*ad.array(np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1))
#    rho.field[indices] += G

#primal = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
#
#def objective(fields):
#    rho, rhoU, rhoE = fields
#    patch = 'obstacle'
#    bc = rhoE.BC
#    start, end = bc[patch].startFace, bc[patch].endFace
#    areas = rhoE.mesh.areas[start:end]
#    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
#    field = rhoE.field[start:end]
#    return ad.sum(field*areas)/(nSteps + 1)
#
#def perturb(fields):
#    rho, rhoU, rhoE = fields
#    patch = 'inlet'
#    bc = rhoU.BC
#    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
#    rhoU.field[start:end][:,0] += 0.1
#
#
#
primal = Solver('tests/cylinder/', {'R': 8.314, 'Cp': 1006, 'gamma': 1.4, 'mu': 2.5e-5, 'Pr': 0.7, 'CFL': 0.2})

def objective(fields):
    rho, rhoU, rhoE = fields
    mesh = rhoE.mesh
    patchID = 'cylinder'
    patch = rhoE.BC[patchID]
    start, end = patch.startFace, patch.endFace
    areas = mesh.areas[start:end]
    nx = mesh.normals[start:end, 0]
    start, end = patch.cellStartFace, patch.cellEndFace
    p = rhoE.field[start:end]*(primal.gamma-1)
    deltas = np.linalg.norm(mesh.cellCentres[start:end]-mesh.cellCentres[patch.internalIndices], axis=1).reshape(-1,1)
    mungUx = primal.mu*(rhoU.field[start:end, 0]/rho.field[start:end]-rhoU.field[patch.internalIndices, 0]/rho.field[patch.internalIndices])/deltas
    return ad.sum((p*nx-mungUx)*areas)/(nSteps + 1)

def perturb(fields):
    rho, rhoU, rhoE = fields
    patch = 'inlet'
    bc = rhoU.BC
    start, end = bc[patch].cellStartFace, bc[patch].cellEndFace
    rhoU.field[start:end][:,0] += 0.1

startTime = 2.0
dt = 1

if __name__ == "__main__":
    mesh = primal.mesh
    option = sys.argv[1]

    if option == 'test':
        eps = ad.array(1E-6)
        mid = np.array([0.5, 0.5, 0.5])

        t = startTime
        primal.dt = dt
        #initialize
        p = CellField.read('p', mesh, t)
        T = CellField.read('T', mesh, t)
        U = CellField.read('U', mesh, t)
        primal.p, primal.T, primal.U = p, T, U
        fields = primal.conservative(U, T, p)
        oldFields = strip(fields)
        oldFields0 = strip(fields)
        fields = explicit(primal.equation, primal.boundary, oldFields, primal)
        J0 = objective(fields)

        G = ad.zeros(oldFields[0].field.shape)
        indices = range(0, mesh.nInternalCells)
        G_subset1 = ad.zeros(G[indices].shape)
        oldFields[0].field[indices] += G_subset1
        #file('old.dot', 'w').write(ad.dot(oldFields[0].field))
        J0 = objective(oldFields)
        fields = explicit(primal.equation, primal.boundary, oldFields, primal)
        J0 += objective(fields)

        #file('old0.dot', 'w').write(ad.dot(oldFields0[0].field))
        J00 = objective(oldFields0)
        fields = explicit(primal.equation, primal.boundary, oldFields0, primal)
        J00 += objective(fields)

        Jt1 = objective(oldFields0)
        fields = explicit(primal.equation, primal.boundary, oldFields0, primal)
        newFields = strip(fields)
        Jt2 = objective(newFields)
        adjt = derivative(Jt2, newFields)
        stackedFields = ad.hstack([phi.field for phi in fields])
        prod = ad.sum(stackedFields*adjt)
        j = derivative(prod, oldFields0)
        s = derivative(Jt1, oldFields0)
        print(j.min(), j.max())
        print(s.min(), s.max())
        adj2 = j + s

        # adjRho_subset = J0.diff(G_subset).toarray().ravel()
        adjRho_subset1 = J0.diff(G_subset1).toarray().ravel()
        adj = derivative(J00, oldFields0)
        adjRho = J0.diff(G).toarray().ravel()
        zeroFields = [CellField.zeros(name, mesh, dimension) for name, dimension in zip(primal.names, primal.dimensions)]
        perturb(zeroFields, 1)
        stackedZeroFields = np.hstack([ad.value(phi.field) for phi in zeroFields])

        eps = ad.array(1E-6)
        G = eps*ad.array(np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1))
        oldFields[0].field[indices] += G

        J1 = objective(oldFields)
        fields = explicit(primal.equation, primal.boundary, oldFields, primal)
        J1 += objective(fields)

        print('J0: ', J0, 'J1: ', J1)
        print('finite difference:', ad.value((J1 - J0) / eps))
        print('automatic diff adjoint:', J1.diff(eps, 'adjoint').toarray())
        print('automatic diff tangent:', J1.diff(eps, 'tangent').toarray())
        print('semi-automatic diff:', np.dot(J1.diff(G).toarray().ravel(), ad.value(G).ravel()) / ad.value(eps))
        # print('semi-automatic diff:', np.dot(adjRho_subset, ad.value(zeroFields[0].field[indices]).ravel()))
        print('semi-automatic diff:', np.dot(adjRho_subset1, ad.value(zeroFields[0].field[indices]).ravel()))
        print('semi-automatic diff:', np.sum(adj*stackedZeroFields))
        print('semi-automatic diff:', np.sum(-adj2*stackedZeroFields))

    elif option == 'orig':
        timeSteps, result = primal.run([startTime, dt], nSteps, writeInterval=writeInterval, objective=objective)
        np.savetxt(mesh.case + '/{0}.{1}.txt'.format(nSteps, writeInterval), timeSteps)
    elif option == 'perturb':
        timeSteps, result = primal.run([startTime, dt], nSteps, objective=objective, perturb=perturb)
    elif option == 'adjoint':
        adjointFields = [CellField.read('{0}a'.format(name), mesh, startTime) for name in primal.names]
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

