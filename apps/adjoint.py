#!/usr/bin/python2
from __future__ import print_function

from adFVM import config, parallel
from adFVM.config import ad
from adFVM.parallel import pprint
from adFVM.field import IOField, Field
from adFVM.matop_petsc import laplacian, ddt
from adFVM.interp import central
from adFVM.memory import printMemUsage
from adFVM.postpro import getAdjointNorm, computeGradients, getAdjointEnergy
from adFVM.solver import Solver

from problem import primal, nSteps, writeInterval, objectiveGradient, perturb, writeResult, nPerturb

import numpy as np
import time
import sys
import os
import cPickle as pkl
import argparse

class Adjoint(Solver):
    def __init__(self, primal, scaling):
        self.primal = primal
        self.scaling = scaling
        self.mesh = primal.mesh
        self.statusFile = primal.statusFile
        self.names = [name + 'a' for name in primal.names]
        self.dimensions = primal.dimensions
        return

    def createFields(self):
        fields = []
        for name, dims in zip(self.names, self.dimensions):
            phi = np.zeros((self.mesh.origMesh.nInternalCells, dims[0]), config.precision)
            fields.append(IOField(name, phi, dims, self.mesh.calculatedBoundary))
        self.fields = fields

    def compile(self):
        primal = self.primal

        self.compileInit(functionName='adjoint_init')

        primal.compile(adjoint=True)
        self.map = primal.adjoint

        if self.scaling:
            self.computer = computeGradients(primal)
        return

    def viscosity(self, solution):
        rho = Field('rho', solution[:,[0]], (1,))
        rhoU = Field('rhoU', solution[:,1:4], (3,))
        rhoE = Field('rhoE', solution[:,[4]], (1,))
        U, T, p = primal.primitive(rho, rhoU, rhoE)
        outputs = self.computer(U.field, T.field, p.field)
        M_2norm = getAdjointNorm(rho, rhoU, rhoE, U, T, p, *outputs)[0]
        M_2normScale = max(parallel.max(M_2norm.field), abs(parallel.min(M_2norm.field)))
        viscosityScale = float(self.scaling)
        pprint('M_2norm: ' +  str(M_2normScale))
        return M_2norm*(viscosityScale/M_2normScale)

    def initPrimalData(self):
        if parallel.mpi.bcast(os.path.exists(primal.statusFile), root=0):
            self.firstCheckpoint, self.result  = primal.readStatusFile()
            pprint('Read status file, checkpoint =', firstCheckpoint)
        else:
            self.firstCheckpoint = 0
            self.result = [0.]*nPerturb
        if parallel.rank == 0:
            self.timeSteps = np.loadtxt(primal.timeStepFile, ndmin=2)
            self.timeSteps = np.concatenate((self.timeSteps, np.array([[np.sum(self.timeSteps[-1]).round(9), 0]])))
        else:
            self.timeSteps = np.zeros((nSteps + 1, 2))
        parallel.mpi.Bcast(self.timeSteps, root=0)
        return
    
    def run(self):
        primal, mesh = self.primal, self.mesh
        result, firstCheckpoint = self.result, self.firstCheckpoint
        timeSteps = self.timeSteps

        startTime = timeSteps[nSteps - firstCheckpoint*writeInterval][0]
        if firstCheckpoint == 0:
            fields = self.initFields(startTime, fields=self.fields)
        else:
            fields = self.initFields(startTime)
        stackedFields = self.stackFields(fields, np)

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
            pprint('Time marching for', ' '.join(self.names))

            if checkpoint == 0:
                t, dt = timeSteps[-1]
                if primal.dynamicMesh:
                    lastMesh, lastSolution = solutions[-1]
                    mesh.origMesh.boundary = lastMesh.boundarydata[m:].reshape(-1,1)
                else:
                    lastSolution = solutions[-1]
                stackedFields  = np.ascontiguousarray(objectiveGradient(lastSolution)/(nSteps + 1))
                fields = self.unstackFields(stackedFields, IOField)
                for phi in fields:
                    phi.info()
                pprint('Adjoint Energy Norm: ', getAdjointEnergy(primal, *fields))
                self.writeFields(fields, t)

            for step in range(0, writeInterval):
                printMemUsage()
                start = time.time()
                fields = self.unstackFields(stackedFields, IOField)
                for phi in fields:
                    phi.info()
                pprint('Adjoint Energy Norm: ', getAdjointEnergy(primal, *fields))

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
                gradients = self.map(previousSolution, stackedFields, dt, t)
                gradient = gradients[0]
                sourceGradient = gradients[1:]
                stackedFields = np.ascontiguousarray(gradient) + np.ascontiguousarray(objectiveGradient(previousSolution)/(nSteps + 1))

                if self.scaling:
                    pprint('Smoothing adjoint field')
                    #weight = adjointViscosity(previousSolution).field
                    #stackedAdjointFields[:mesh.origMesh.nInternalCells] += dt*adjointSmoother(stackedAdjointFields, weight)
                    stackedPhi = Field('a', stackedFields, (5,))
                    stackedPhi.old = stackedFields
                    start2 = time.time() 
                    weight = central(self.viscosity(previousSolution), mesh.origMesh)
                    start3 = time.time()
                    #stackedAdjointFields[:mesh.origMesh.nLocalCells] = BCs(stackedPhi, ddt(stackedPhi, dt) - laplacian(stackedPhi, weight)).solve()
                    stackedFields[:mesh.origMesh.nInternalCells] = (ddt(stackedPhi, dt) - laplacian(stackedPhi, weight)).solve()
                    start4 = time.time()
                    pprint('Timers 1:', start3-start2, '2:', start4-start3)

                # compute sensitivity using adjoint solution
                for index, perturbation in enumerate(perturb):
                    for derivative, delphi in zip(sourceGradient, perturbation(None, mesh.origMesh, t)):
                        result[index] += np.sum(np.ascontiguousarray(derivative) * delphi)

                parallel.mpi.Barrier()
                end = time.time()
                pprint('Time for adjoint iteration: {0}'.format(end-start))
                pprint('Time since beginning:', end-config.runtime)
                pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

            #exit(1)
            fields = self.unstackFields(stackedFields, IOField)
            self.writeFields(fields, t)
            self.writeStatusFile([checkpoint + 1, result])

        for index in range(0, nPerturb):
            writeResult('adjoint', result[index], '{} {}'.format(index, self.scaling))
        self.removeStatusFile()
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scaling', required=False)
    user, args = parser.parse_known_args()

    adjoint = Adjoint(primal, user.scaling)
    adjoint.initPrimalData()

    adjoint.createFields()
    primal.readFields(adjoint.timeSteps[nSteps-writeInterval][0])
    adjoint.compile()

    adjoint.run()

if __name__ == '__main__':
    main()

