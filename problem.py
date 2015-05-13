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

#primal = RCF('cases/convection/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
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

#primal = RCF('cases/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0., 'Pr': 0.7, 'CFL': 0.2})
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
#primal = RCF('cases/cylinder/', mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)))
#def objective(fields):
#    rho, rhoU, rhoE = fields
#    mesh = rho.mesh
#    patchID = 'cylinder'
#    patch = mesh.boundary[patchID]
#    nF = patch['nFaces']
#    start, end = patch['startFace'], patch['startFace'] + nF
#    areas = mesh.areas[start:end]
#    nx = mesh.normals[start:end, 0].reshape((-1, 1))
#    cellStartFace = mesh.nInternalCells + start - mesh.nInternalFaces
#    cellEndFace = mesh.nInternalCells + end - mesh.nInternalFaces
#    internalIndices = mesh.owner[start:end]
#    start, end = cellStartFace, cellEndFace
#    p = rhoE.field[start:end]*(primal.gamma-1)
#    #deltas = (mesh.cellCentres[start:end]-mesh.cellCentres[internalIndices]).norm(2, axis=1, keepdims=True)
#    deltas = (mesh.cellCentres[start:end]-mesh.cellCentres[internalIndices]).norm(2, axis=1).reshape((nF,1))
#    T = rhoE/(rho*primal.Cv)
#    #mungUx = (rhoU.field[start:end, [0]]/rho.field[start:end]-rhoU.field[internalIndices, [0]]/rho.field[internalIndices])*primal.mu(T).field[start:end]/deltas
#    mungUx = (rhoU.field[start:end, 0].reshape((nF,1))/rho.field[start:end]-rhoU.field[internalIndices, 0].reshape((nF,1))/rho.field[internalIndices])*primal.mu(T).field[start:end]/deltas
#    return ad.sum((p*nx-mungUx)*areas)/(nSteps + 1)
#
#def perturb(stackedFields, t):
#    mesh = primal.mesh.origMesh
#    mid = np.array([-0.0032, 0.0, 0.])
#    #G = 1e-6*np.exp(-1e7*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
#    G = 1e-4*np.exp(-1e2*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
#    #rho
#    if t == startTime:
#        stackedFields[:mesh.nInternalCells, 0] += G
#        stackedFields[:mesh.nInternalCells, 1] += G*100
#        stackedFields[:mesh.nInternalCells, 4] += G*2e5

#primal = RCF('/home/talnikar/foam/blade/les/')
primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')
def objective(fields):
    rho, rhoU, rhoE = fields
    solver = rhoE.solver
    mesh = rhoE.mesh

    res = 0
    for patchID in ['suction', 'pressure']:
        patch = rhoE.BC[patchID]
        start, end = patch.startFace, patch.endFace
        cellStart, cellEnd = patch.cellStartFace, patch.cellEndFace
        areas = mesh.areas[start:end]
        U, T, p = solver.primitive(rho, rhoU, rhoE)
        
        Ti = T.field[mesh.owner[start:end]] 
        Tw = 300*Ti/Ti
        deltas = (mesh.cellCentres[cellStart:cellEnd]-mesh.cellCentres[patch.internalIndices]).norm(2, axis=1).reshape((end-start, 1))
        dtdn = (Tw-Ti)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        dT = 120
        res += ad.sum(k*dtdn*areas)/(dT*ad.sum(areas)*(nSteps + 1) + config.VSMALL)
    return res

def perturb(stackedFields, t):
    mesh = primal.mesh.origMesh
    mid = np.array([-0.08, 0.014, 0.005])
    G = 1e-3*np.exp(-1e5*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #G = 1e-4*np.exp(-1e2*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    if t == startTime:
        stackedFields[:mesh.nInternalCells, 0] += G
        stackedFields[:mesh.nInternalCells, 1] += G*100
        stackedFields[:mesh.nInternalCells, 4] += G*2e5

nSteps = 20000
writeInterval = 100
startTime = 2.0
dt = 1e-9

pprint('Compiling objective')
stackedFields = ad.matrix()
stackedFields.tag.test_value = np.random.rand(primal.mesh.origMesh.nCells, 5).astype(config.precision)
fields = primal.unstackFields(stackedFields, CellField)
objectiveValue = objective(fields)
objectiveFunction = T.function([stackedFields], objectiveValue, mode=config.compile_Mode)
# objective is anyways going to be a sum over all processors
# so no additional code req to handle parallel case
objectiveGradient = T.function([stackedFields], ad.grad(objectiveValue, stackedFields), mode=config.compile_mode)

def writeResult(option, result):
    resultFile = primal.mesh.case + '/objective.txt'
    globalResult = parallel.sum(result)
    if parallel.rank == 0:
        if option == 'perturb':
            previousResult = float(open(resultFile).readline().split(' ')[1])
            globalResult -= previousResult
        f = open(resultFile, 'a')
        f.write('{0} {1}\n'.format(option, globalResult))
        f.close()

if __name__ == "__main__":
    mesh = primal.mesh.origMesh
    option = sys.argv[1]
    timeStepFile = primal.mesh.case + '/{0}.{1}.txt'.format(nSteps, writeInterval)
    
    if option == 'orig':
        perturb = None
        dts = dt

    elif option == 'perturb':
        writeInterval = config.LARGE
        if parallel.rank == 0:
            timeSteps = np.loadtxt(timeStepFile)
            timeSteps = np.concatenate((timeSteps, np.array([[0, 0]])))
        else:
            timeSteps = np.zeros((nSteps+1, 2))
        parallel.mpi.Bcast(timeSteps, root=0)
        dts = timeSteps[:,1]

    elif option == 'test':
        primal.initFields(startTime)
        a = np.zeros((mesh.nCells, 5))
        perturb(a, startTime)
        fields = primal.unstackFields(a, IOField)
        primal.writeFields(fields, 100.0)

        #p = np.zeros((mesh.nCells, 5))
        #fields = primal.initFields(startTime)
        #stackedFields = primal.stackFields(fields, np) 
        #result = objectiveFunction(stackedFields)
        #perturb(p, startTime)
        #print(np.sum(objectiveGradient(stackedFields)*p))
        #stackedFields += p
        #resultp = objectiveFunction(stackedFields)
        #print(resultp-result)

        #primal.adjoint = True
        #p = np.zeros((mesh.nCells, 5))
        #perturb(p, startTime)
        #writeInterval = config.LARGE
        #timeSteps, result = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, objective=objectiveFunction, perturb=None)
        #timeSteps, resultp = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, objective=objectiveFunction, perturb=perturb)
        #solutions = primal.run(startTime=startTime, dt=dt, nSteps=nSteps, writeInterval=writeInterval, mode='forward')
        #grad1 = np.ascontiguousarray(objectiveGradient(solutions[-1]))
        #grad2 = np.ascontiguousarray(primal.gradient(solutions[0], grad1))
        #grad3 = np.ascontiguousarray(objectiveGradient(solutions[0]))
        #grad = grad2 + grad3
        #print(np.sum(grad*p))
        #print(resultp-result)

        exit()
    else:
        print('WTF')
        exit()

    timeSteps, result = primal.run(startTime=startTime, dt=dts, nSteps=nSteps, writeInterval=writeInterval, objective=objectiveFunction, perturb=perturb)
    writeResult(option, result)
    if option == 'orig' and parallel.rank == 0:
            np.savetxt(timeStepFile, timeSteps)
