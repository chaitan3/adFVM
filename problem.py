#!/usr/bin/python2
from __future__ import print_function

import numpy as np
import sys

import config, parallel
from config import ad, T
from parallel import pprint
from pyRCF import RCF
from solver import euler as explicit
from field import CellField, Field, IOField

#primal = RCF('tests/convection/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
#
#def objective(fields):
#    rho, rhoU, rhoE = fields
#    mesh = rho.mesh
#    mid = np.array([0.75, 0.5, 0.5])
#    indices = range(0, mesh.nInternalCells)
#    G = np.exp(-100*config.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1)*mesh.volumes[indices]
#    return ad.sum(rho.field[indices]*G)/(nSteps + 1)
#
#def perturb(fields, eps=1E-2):
#    rho, rhoU, rhoE = fields
#    mesh = rho.mesh
#    mid = np.array([0.5, 0.5, 0.5])
#    indices = range(0, mesh.nInternalCells)
#    G = eps*ad.array(np.exp(-100*config.norm(mid-mesh.cellCentres[indices], axis=1)**2).reshape(-1,1))
#    rho.field[indices] += G

#primal = RCF('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
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
#primal = RCF('tests/cylinder/', mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)))
primal = RCF('tests/cylinder/', CFL=0.2, timeIntegrator='euler', mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)))

def objective(fields):
    rho, rhoU, rhoE = fields
    mesh = rho.mesh
    patchID = 'cylinder'
    patch = mesh.boundary[patchID]
    nF = patch['nFaces']
    start, end = patch['startFace'], patch['startFace'] + nF
    areas = mesh.areas[start:end]
    nx = mesh.normals[start:end, 0].reshape((-1, 1))
    cellStartFace = mesh.nInternalCells + start - mesh.nInternalFaces
    cellEndFace = mesh.nInternalCells + end - mesh.nInternalFaces
    internalIndices = mesh.owner[start:end]
    start, end = cellStartFace, cellEndFace
    p = rhoE.field[start:end]*(primal.gamma-1)
    deltas = (mesh.cellCentres[start:end]-mesh.cellCentres[internalIndices]).norm(2, axis=1).reshape((nF,1))
    T = rhoE/(rho*primal.Cv)
    mungUx = (rhoU.field[start:end, 0].reshape((nF,1))/rho.field[start:end]-rhoU.field[internalIndices, 0].reshape((nF,1))/rho.field[internalIndices])*primal.mu(T).field[start:end]/deltas
    return ad.sum((p*nx-mungUx)*areas)/(nSteps + 1)

def perturb(stackedFields, t):
    mesh = primal.mesh.origMesh
    mid = np.array([-0.0032, 0.0, 0.])
    G = 1e-3*np.exp(-1e7*config.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    if t == startTime:
        stackedFields[:mesh.nInternalCells, 0] += G
        stackedFields[:mesh.nInternalCells, 1] += G*100
        stackedFields[:mesh.nInternalCells, 4] += G*2e5

#primal = RCF('/home/talnikar/foam/blade/laminar/', CFL=0.6)
#def objective(fields):
#    rho, rhoU, rhoE = fields
#    solver = rhoE.solver
#    mesh = rhoE.mesh
#    patchID = 'suction'
#    patch = rhoE.BC[patchID]
#    start, end = patch.startFace, patch.endFace
#    areas = mesh.areas[start:end]
#    U, T, p = solver.primitive(rho, rhoU, rhoE)
#    
#    Ti = T.field[mesh.owner[start:end]] 
#    Tw = 300*Ti/Ti
#    deltas = config.norm(mesh.cellCentres[start:end]-mesh.cellCentres[patch.internalIndices], axis=1).reshape(-1,1)
#    dtdn = (Tw-Ti)/deltas
#    k = solver.Cp*solver.mu(Tw)/solver.Pr
#    dT = 120
#    return ad.sum(k*dtdn*areas)/(dT*ad.sum(areas)*(nSteps + 1))

nSteps = 20000
writeInterval = 100
startTime = 2.0
dt = 1e-9
#nSteps = 10
#writeInterval = 2
#startTime = 2.0
#dt = 1e-9

pprint('Compiling objective')
stackedFields = ad.matrix()
stackedFields.tag.test_value = np.random.rand(primal.mesh.origMesh.nCells, 5).astype(config.precision)
fields = primal.unstackFields(stackedFields, CellField)
objectiveValue = objective(fields)
objectiveFunction = T.function([stackedFields], objectiveValue)
# objective is anyways going to be a sum over all processors
# so no additional code req to handle parallel case
objectiveGradient = T.function([stackedFields], ad.grad(objectiveValue, stackedFields))

def writeResult(option, result):
    mesh = primal.mesh.origMesh
    globalResult = parallel.sum(result)
    if parallel.rank == 0:
        f = open(mesh.case + '/objective.txt', 'a')
        f.write('{0} {1}\n'.format(option, globalResult))
        f.close()

if __name__ == "__main__":
    mesh = primal.mesh.origMesh
    option = sys.argv[1]
    
    if option == 'orig':
        perturb = None
    elif option == 'perturb':
        writeInterval = config.LARGE
    elif option == 'test':
        primal.initFields(startTime)
        a = np.zeros((mesh.nCells, 5))
        perturb(a, startTime)
        fields = primal.unstackFields(a, IOField)
        primal.writeFields(fields, 100.0)
        exit()
    else:
        print('WTF')
        exit()

    timeSteps, result = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, objective=objectiveFunction, perturb=perturb)
    writeResult(option, result)
    if option == 'orig' and parallel.rank == 0:
            np.savetxt(mesh.case + '/{0}.{1}.txt'.format(nSteps, writeInterval), timeSteps)
