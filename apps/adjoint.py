#!/usr/bin/python2
from __future__ import print_function

from adFVM import config, parallel
from adFVM.config import ad
from adFVM.parallel import pprint
from adFVM.field import IOField, Field
from adFVM.matop_petsc import laplacian, ddt
#from adFVM.matop import laplacian, ddt
from adFVM.interp import central
from adFVM.memory import printMemUsage
from adFVM.postpro import getAdjointNorm, computeGradients, getAdjointEnergy
from adFVM.solver import Solver

from problem import primal, nSteps, writeInterval, reportInterval, objectiveGradient, perturb, writeResult, nPerturb, parameters

import numpy as np
import time
import os
import argparse


class Adjoint(Solver):
    def __init__(self, primal, scaling):
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
        self.compileInit(functionName='adjoint_init')

        if self.scaling:
            self.computer = computeGradients(primal)

        primal.compile(adjoint=self)
        self.map = primal.adjoint

        return

    def getGradFields(self):
        variables = []
        for param in parameters:
            if param == 'source':
                pprint('Gradient wrt source')
                variables.extend(list(zip(*primal.sourceTerms)[0]))
            elif param == 'mesh':
                pprint('Gradient wrt mesh')
                variables.extend([getattr(self.mesh, field) for field in Mesh.gradFields])
            elif isinstance(param, tuple):
                assert param[0] == 'BCs'
                pprint('Gradient wrt', param)
                _, phi, patchID, key = param
                patch = getattr(primal, phi).phi.BC[patchID]
                index = patch.keys.index(key)
                variables.append(patch.inputs[index][0])
            elif isinstance(param, ad.TensorType):
                variables.append(param)
        return variables

    def viscosity(self, rho, rhoU, rhoE):
        U, T, p = primal.primitive(rho, rhoU, rhoE)
        outputs = self.computer(U, T, p)
        M_2norm = getAdjointNorm(rho, rhoU, rhoE, U, T, p, *outputs)[0]
        M_2normScale = max(parallel.max(M_2norm.field), abs(parallel.min(M_2norm.field)))
        viscosityScale = float(self.scaling)
        pprint('M_2norm: ' +  str(M_2normScale))
        return M_2norm*(viscosityScale/M_2normScale)

    def initPrimalData(self):
        if parallel.mpi.bcast(os.path.exists(primal.statusFile), root=0):
            self.firstCheckpoint, self.result  = primal.readStatusFile()
            pprint('Read status file, checkpoint =', self.firstCheckpoint)
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
        mesh = self.mesh
        result, firstCheckpoint = self.result, self.firstCheckpoint
        timeSteps = self.timeSteps

        startTime = timeSteps[nSteps - firstCheckpoint*writeInterval][0]
        if firstCheckpoint == 0:
            fields = self.initFields(startTime, read=False)
        else:
            fields = self.initFields(startTime)

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
                fields = objectiveGradient(*lastSolution)
                fields = [phi/(nSteps + 1) for phi in fields]
                fields = self.getFields(fields, IOField)
                #for phi in fields:
                #    phi.info()
                #pprint('Adjoint Energy Norm: ', getAdjointEnergy(primal, *fields))
                self.writeFields(fields, t, skipProcessor=True)

            for step in range(0, writeInterval):
                report = (step % reportInterval) == 0
                
                adjointIndex = writeInterval-1 - step
                t, dt = timeSteps[primalIndex + adjointIndex]
                if primal.dynamicMesh:
                    previousMesh, previousSolution = solutions[adjointIndex]
                    # new mesh boundary
                    mesh.origMesh.boundary = previousMesh.boundary
                else:
                    previousSolution = solutions[adjointIndex]

                if report:
                    printMemUsage()
                    start = time.time()
                    for phi in fields:
                        phi.info()
                    pprint('Adjoint Energy Norm: ', getAdjointEnergy(primal, *fields))
                    pprint('Time step', adjointIndex)

                inputs = previousSolution + fields + [dt, t]
                outputs = self.map(*inputs)
                gradient = outputs[:len(fields)]
                paramGradient = outputs[len(fields):]
                objGradient = objectiveGradient(*previousSolution)
                objGradient = [phi/(nSteps + 1) for phi in objGradient]
                for index in range(0, len(fields)):
                    fields[index].field = gradient[index] + objGradient[index]

                if self.scaling:
                    if report:
                        pprint('Smoothing adjoint field')
                    nInternalCells = mesh.origMesh.nInternalCells
                    start2 = time.time() 
                    weight = central(self.viscosity(*previousSolution), mesh.origMesh)
                    start3 = time.time()

                    stackedFields = np.concatenate([phi.field for phi in fields], axis=1)
                    stackedFields = np.ascontiguousarray(stackedFields)
                    stackedPhi = Field('a', stackedFields, (5,))
                    stackedPhi.old = stackedFields
                    newStackedFields = (ddt(stackedPhi, dt) - laplacian(stackedPhi, weight)).solve()
                    fields[0].field[:nInternalCells] = newStackedFields[:, [0]]
                    fields[1].field[:nInternalCells] = newStackedFields[:, [1,2,3]]
                    fields[2].field[:nInternalCells] = newStackedFields[:, [4]]
                    for phi in fields:
                        phi.field = np.ascontiguousarray(phi.field)

                    start4 = time.time()
                    pprint('Timers 1:', start3-start2, '2:', start4-start3)

                # compute sensitivity using adjoint solution
                for index in range(0, len(perturb)):
                    perturbation = perturb[index](None, mesh.origMesh, t)
                    #if not isinstance(perturbation, list) or (len(parameters) == 1 and len(perturbation) > 1):
                    #    perturbation = [perturbation]
                    for derivative, delphi in zip(paramGradient, perturbation):
                        result[index] += np.sum(derivative * delphi)

                #parallel.mpi.Barrier()
                if report:
                    end = time.time()
                    pprint('Time for adjoint iteration: {0}'.format(end-start))
                    pprint('Time since beginning:', end-config.runtime)
                    pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

            #exit(1)
            self.writeFields(fields, t, skipProcessor=True)
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

