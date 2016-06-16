#!/usr/bin/python2
from __future__ import print_function

import config, parallel
from config import ad
from parallel import pprint
from field import IOField, Field
#from op import laplacian
from matop_petsc import laplacian, ddt
#from matop import laplacian, ddt
from interp import central
from problem import primal, nSteps, writeInterval, objectiveGradient, perturb, writeResult
from problem import nPerturb
from compat import printMemUsage
from compute import getAdjointNorm, computeFields, getAdjointEnergy

import numpy as np
import time
import sys
import os
import cPickle as pkl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--smooth', action='store_true')
parser.add_argument('--scaling', required=False, default=0.0)
user, args = parser.parse_known_args()

primal.adjoint = True
mesh = primal.mesh
if parallel.mpi.bcast(os.path.exists(primal.statusFile), root=0):
    firstCheckpoint, result  = primal.readStatusFile()
    pprint('Read status file, checkpoint =', firstCheckpoint)
else:
    firstCheckpoint = 0
    result = [0.]*nPerturb
if parallel.rank == 0:
    timeSteps = np.loadtxt(primal.timeStepFile, ndmin=2)
    timeSteps = np.concatenate((timeSteps, np.array([[np.sum(timeSteps[-1]).round(9), 0]])))
else:
    timeSteps = np.zeros((nSteps + 1, 2))
parallel.mpi.Bcast(timeSteps, root=0)

# provision for restaring adjoint
adjointNames = ['{0}a'.format(name) for name in primal.names]
if firstCheckpoint == 0:
    adjointFields = [IOField(name, np.zeros((mesh.origMesh.nInternalCells, dimensions[0]), config.precision), dimensions, mesh.calculatedBoundary) for name, dimensions in zip(adjointNames, primal.dimensions)]
else:
    t = timeSteps[nSteps - firstCheckpoint*writeInterval][0]
    IOField.openHandle(t)
    adjointFields = [IOField.read(name) for name in adjointNames]
    IOField.closeHandle()

adjointInternalFields = [phi.complete() for phi in adjointFields]
adjointNewFields = [phi.phi.field for phi in adjointFields]
# UGLY HACK
oldFunc = primal.getBCFields
primal.getBCFields = lambda: adjointFields
adjointInitFunc = primal.function(adjointInternalFields, adjointNewFields, 'adjoint_init')
primal.getBCFields = oldFunc

# dummy initialize
primal.initialize(timeSteps[nSteps-writeInterval][0])
if user.smooth:
    computer = computeFields(primal)
primal.compile()

newFields = adjointInitFunc(*[phi.field for phi in adjointFields])
for phi, field in zip(adjointFields, newFields):
    phi.field = field

def unstackAdjointFields(stackedAdjointFields):
    return primal.unstackFields(stackedAdjointFields, IOField, names=adjointNames, boundary=mesh.calculatedBoundary)

def writeAdjointFields(stackedAdjointFields, writeTime):
    # TODO: fix unstacking F_CONTIGUOUS
    start = time.time()
    IOField.openHandle(writeTime)
    for phi in unstackAdjointFields(stackedAdjointFields):
        phi.field = np.ascontiguousarray(phi.field)
        phi.write()
    IOField.closeHandle()
    parallel.mpi.Barrier()
    end = time.time()
    pprint('Time for writing fields: {0}'.format(end-start))
    pprint()

def adjointViscosity(solution):
    rho = Field('rho', solution[:,[0]], (1,))
    rhoU = Field('rhoU', solution[:,1:4], (3,))
    rhoE = Field('rhoE', solution[:,[4]], (1,))
    U, T, p = primal.primitive(rho, rhoU, rhoE)
    SF = primal.stackFields([p, U, T], np)
    outputs = computer(SF)
    M_2norm = getAdjointNorm(rho, rhoU, rhoE, U, T, p, *outputs)[0]
    M_2normScale = max(parallel.max(M_2norm.field), abs(parallel.min(M_2norm.field)))
    viscosityScale = float(user.scaling)
    #print(parallel.rank, M_2normScale)
    return M_2norm*(viscosityScale/M_2normScale)

# adjont field smoothing,
#if user.smooth:
#    adjointField = Field('a', ad.matrix(), (5,))
#    weight = Field('w', ad.bcmatrix(), (1,))
#    smoother = laplacian(adjointField, central(weight, primal.mesh))
#    adjointSmoother = primal.function([adjointField.field, weight.field], smoother.field, 'smoother', BCs=False)
#    pprint()

# local adjoint fields
stackedAdjointFields = primal.stackFields(adjointFields, np)
pprint('STARTING ADJOINT')
pprint('Number of steps:', nSteps)
pprint('Write interval:', writeInterval)
pprint()

totalCheckpoints = nSteps/writeInterval
for checkpoint in range(firstCheckpoint, totalCheckpoints):
    pprint('PRIMAL FORWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
    primalIndex = nSteps - (checkpoint + 1)*writeInterval
    t, dt = timeSteps[primalIndex]
    #writeInterval = 1
    solutions = primal.run(startTime=t, dt=dt, nSteps=writeInterval, mode='forward')

    pprint('ADJOINT BACKWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
    # if starting from 0, create the adjointField
    pprint('Time marching for', ' '.join(adjointNames))

    if checkpoint == 0:
        t, dt = timeSteps[-1]
        if primal.dynamicMesh:
            lastMesh, lastSolution = solutions[-1]
            mesh.origMesh.boundary = lastMesh.boundarydata[m:].reshape(-1,1)
        else:
            lastSolution = solutions[-1]
        stackedAdjointFields  = np.ascontiguousarray(objectiveGradient(lastSolution)/(nSteps + 1))
        adjointFields = unstackAdjointFields(stackedAdjointFields)
        for phi in adjointFields:
            phi.info()
        pprint('Adjoint Energy Norm: ', getAdjointEnergy(primal, *adjointFields))
        writeAdjointFields(stackedAdjointFields, t)

    for step in range(0, writeInterval):
        printMemUsage()
        start = time.time()
        adjointFields = unstackAdjointFields(stackedAdjointFields)
        for phi in adjointFields:
            phi.info()
        pprint('Adjoint Energy Norm: ', getAdjointEnergy(primal, *adjointFields))

        adjointIndex = writeInterval-1 - step
        pprint('Time step', adjointIndex)
        t, dt = timeSteps[primalIndex + adjointIndex]
        if primal.dynamicMesh:
            previousMesh, previousSolution = solutions[adjointIndex]
            # new mesh boundary
            mesh.origMesh.boundary = previousMesh.boundary
        else:
            previousSolution = solutions[adjointIndex]
        #paddedPreviousSolution = parallel.getRemoteCells(previousSolution, mesh)
        ## adjoint time stepping
        #paddedJacobian = np.ascontiguousarray(primal.gradient(paddedPreviousSolution, stackedAdjointFields))
        #jacobian = parallel.getAdjointRemoteCells(paddedJacobian, mesh)
        gradients = primal.gradient(previousSolution, stackedAdjointFields, dt, t)
        gradient = gradients[0]
        sourceGradient = gradients[1:]
        stackedAdjointFields = np.ascontiguousarray(gradient) + np.ascontiguousarray(objectiveGradient(previousSolution)/(nSteps + 1))

        if user.smooth:
            pprint('Smoothing adjoint field')
            #weight = adjointViscosity(previousSolution).field
            #stackedAdjointFields[:mesh.origMesh.nInternalCells] += dt*adjointSmoother(stackedAdjointFields, weight)
            stackedPhi = Field('a', stackedAdjointFields, (5,))
            stackedPhi.old = stackedAdjointFields
            start2 = time.time() 
            weight = central(adjointViscosity(previousSolution), mesh.origMesh)
            start3 = time.time()
            #stackedAdjointFields[:mesh.origMesh.nLocalCells] = BCs(stackedPhi, ddt(stackedPhi, dt) - laplacian(stackedPhi, weight)).solve()
            stackedAdjointFields[:mesh.origMesh.nInternalCells] = (ddt(stackedPhi, dt) - laplacian(stackedPhi, weight)).solve()
            start4 = time.time()
            pprint('Timers 1:', start3-start2, '2:', start4-start3)

        # compute sensitivity using adjoint solution
        for index, perturbation in enumerate(perturb):
            for derivative, delphi in zip(sourceGradient, perturbation(mesh.origMesh)):
                result[index] += np.sum(np.ascontiguousarray(derivative) * delphi)

        parallel.mpi.Barrier()
        end = time.time()
        pprint('Time for adjoint iteration: {0}'.format(end-start))
        pprint('Time since beginning:', end-config.runtime)
        pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

    writeAdjointFields(stackedAdjointFields, t)
    primal.writeStatusFile([checkpoint + 1, result])

for index in range(0, nPerturb):
    writeResult('adjoint', result[index], '{} {}'.format(index, user.scaling))
if parallel.rank == 0:
    os.remove(primal.statusFile)
