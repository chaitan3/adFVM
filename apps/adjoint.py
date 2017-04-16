#!/usr/bin/python2 -u
from __future__ import print_function

from adFVM import config, parallel
from adFVM.config import ad
from adFVM.parallel import pprint
from adFVM.field import IOField, Field
from adFVM.interp import central
from adFVM.memory import printMemUsage
from adFVM.postpro import getAdjointViscosity, getAdjointEnergy
from adFVM.solver import Solver

from problem import primal, nSteps, writeInterval, reportInterval, perturb, writeResult, nPerturb, parameters, source, adjParams

import numpy as np
import time
import os
import argparse
import adFVMcpp_ad as adFVMcpp

class Adjoint(Solver):
    def __init__(self, primal):
        self.scaling = adjParams[0]
        self.viscosityType = adjParams[1]
        self.viscosityScaler = adjParams[2]
        self.mesh = primal.mesh
        self.statusFile = primal.statusFile
        self.names = [name + 'a' for name in primal.names]
        self.dimensions = primal.dimensions
        self.sensTimeSeriesFile = self.mesh.case + 'sensTimeSeries.txt'
        self.energyTimeSeriesFile = self.mesh.case + 'energyTimeSeries.txt'
        self.firstRun = True
        return

    def initFields(self, fields):
        newFields = primal.adFVMcpp.ghost_default(*[phi.field for phi in fields])
        return self.getFields(newFields, IOField, refFields=fields)

    def createFields(self):
        fields = []
        for name, dims in zip(self.names, self.dimensions):
            phi = np.zeros((self.mesh.origMesh.nInternalCells, dims[0]), config.precision)
            fields.append(IOField(name, phi, dims, self.mesh.defaultBoundary))
        self.fields = fields
        return fields

    def compile(self):
        #self.compileInit(functionName='adjoint_init')
        primal.compile(adjoint=self)
        adFVMcpp.init(*([self.mesh] + [phi.boundary for phi in primal.fields] + [primal.__class__.defaultConfig]))
        primal.adjoint = self
        #self.map = primal.gradient
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

    def initPrimalData(self):
        if parallel.mpi.bcast(os.path.exists(primal.statusFile), root=0):
            self.firstCheckpoint, self.result  = primal.readStatusFile()
            pprint('Read status file, checkpoint =', self.firstCheckpoint)
        else:
            self.firstCheckpoint = 0
            self.result = [0.]*nPerturb
        if parallel.rank == 0:
            self.timeSteps = np.loadtxt(primal.timeStepFile, ndmin=2)
            assert self.timeSteps.shape == (nSteps, 2)
            self.timeSteps = np.concatenate((self.timeSteps, np.array([[np.sum(self.timeSteps[-1]).round(12), 0]])))
        else:
            self.timeSteps = np.zeros((nSteps + 1, 2))
        #print(self.timeSteps.shape)
        parallel.mpi.Bcast(self.timeSteps, root=0)
        return
    
    def run(self, readFields=False):
        mesh = self.mesh
        result, firstCheckpoint = self.result, self.firstCheckpoint
        timeSteps = self.timeSteps
        sensTimeSeries = []
        energyTimeSeries = []

        startTime = timeSteps[nSteps - firstCheckpoint*writeInterval][0]
        if (firstCheckpoint > 0) or readFields:
            fields = self.readFields(startTime)
        else:
            fields = self.createFields()

        pprint('STARTING ADJOINT')
        pprint('Number of steps:', nSteps)
        pprint('Write interval:', writeInterval)
        pprint()

        totalCheckpoints = nSteps/writeInterval
        for checkpoint in range(firstCheckpoint, totalCheckpoints):
            pprint('PRIMAL FORWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
            primalIndex = nSteps - (checkpoint + 1)*writeInterval
            t = timeSteps[primalIndex, 0]
            dts = timeSteps[primalIndex:primalIndex+writeInterval+1, 1]

            #self.fields = fields
            #print(fields[0].field.max())
            solutions = primal.run(startTime=t, dt=dts, nSteps=writeInterval, mode='forward', reportInterval=reportInterval)
            #print(fields[0].field.max())

            pprint('ADJOINT BACKWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
            pprint('Time marching for', ' '.join(self.names))

            if checkpoint == 0:
                t, _ = timeSteps[-1]
                if primal.dynamicMesh:
                    lastMesh, lastSolution = solutions[-1]
                    mesh.origMesh.boundary = lastMesh.boundarydata[m:].reshape(-1,1)
                else:
                    lastSolution = solutions[-1]
                #fields = objectiveGradient(*lastSolution)
                #fields = [phi/(nSteps + 1) for phi in fields]
                #fields = self.getFields(fields, IOField)
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
                primal.updateSource(source(previousSolution, mesh.origMesh, t))

                if report:
                    printMemUsage()
                    start = time.time()
                    for phi in fields:
                        phi.info()
                    energyTimeSeries.append(getAdjointEnergy(primal, *fields))
                    pprint('Time step', adjointIndex)
                else:
                    pprint('Time step', adjointIndex)

                inputs = [phi.field for phi in previousSolution] + \
                         [phi[1] for phi in primal.sourceTerms] + \
                         [phi.field for phi in fields] + [dt, t, nSteps]
                #outputs = self.map(*inputs)
                #print(fields[0].field.max())
                outputs = adFVMcpp.forward(*inputs)
                n = len(fields)
                gradient = outputs

                #print(sum([(1e-3*phi).sum() for phi in gradient]))
                #inp1 = inputs[:3] + inputs[-3:-1]
                #inp2 = [phi + 1e-3 for phi in inputs[:3]] + inputs[-3:-1]
                #x1 = primal.map(*inp1)[-2]
                #x2 = primal.map(*inp2)[-2]
                #print(x1, x2, x1-x2)
                #import pdb;pdb.set_trace()

                gradient, paramGradient = outputs[:n], outputs[n:]
                for index in range(0, len(fields)):
                    fields[index].field = gradient[index]
                #print(fields[0].field.max())

                if self.scaling:
                    if report:
                        pprint('Smoothing adjoint field')
                    nInternalCells = mesh.origMesh.nInternalCells
                    start2 = time.time() 
                    inputs = previousSolution + [self.scaling]
                    kwargs = {'visc': self.viscosityType, 'scale': self.viscosityScaler, 'report':report}
                    weight = central(getAdjointViscosity(*inputs, **kwargs), mesh.origMesh)
                    start3 = time.time()

                    stackedFields = np.concatenate([phi.field for phi in fields], axis=1)
                    stackedFields = np.ascontiguousarray(stackedFields)

                    if self.matop:
                        newStackedFields = adFVMcpp.viscosity(stackedFields, weight.field, dt)
                    else:
                        stackedPhi = Field('a', stackedFields, (5,))
                        stackedPhi.old = stackedFields
                        newStackedFields = (matop_petsc.ddt(stackedPhi, dt) - matop_petsc.laplacian(stackedPhi, weight, correction=False)).solve()
                        #newStackedFields = stackedFields/(1 + weight*dt)

                    newFields = [newStackedFields[:,[0]], 
                                 newStackedFields[:,[1,2,3]], 
                                 newStackedFields[:,[4]]
                                ]
                            
                    fields = self.getFields(newFields, IOField)
                    for phi in fields:
                        phi.field = np.ascontiguousarray(phi.field)

                    start4 = time.time()
                    pprint('Timers 1:', start3-start2, '2:', start4-start3)

                # compute sensitivity using adjoint solution
                sensTimeSeries.append([0.]*nPerturb)
                for index in range(0, len(perturb)):
                    perturbation = perturb[index](None, mesh.origMesh, t)
                    if isinstance(perturbation, tuple):
                        perturbation = list(perturbation)
                    if not isinstance(perturbation, list):# or (len(parameters) == 1 and len(perturbation) > 1):
                        perturbation = [perturbation]
                        # complex parameter perturbation not supported
                    
                    for derivative, delphi in zip(paramGradient, perturbation):
                        sensitivity = np.sum(derivative * delphi)
                        result[index] += sensitivity
                        sensTimeSeries[-1][index] = sensitivity

                #parallel.mpi.Barrier()
                if report:
                    end = time.time()
                    pprint('Time for adjoint iteration: {0}'.format(end-start))
                    pprint('Time since beginning:', end-config.runtime)
                    pprint('Simulation Time and step: {0}, {1}\n'.format(*timeSteps[primalIndex + adjointIndex + 1]))

            #exit(1)
            #print(fields[0].field.max())
            self.writeFields(fields, t, skipProcessor=True)
            #print(fields[0].field.max())
            self.writeStatusFile([checkpoint + 1, result])
            sensTimeSeries = parallel.mpi.gather(sensTimeSeries, root=0)
            #energyTimeSeries = mpi.gather(timeSeries, root=0)
            if parallel.rank == 0:
                sensTimeSeries = np.sum(sensTimeSeries, axis=0)
            if parallel.rank == 0:
                with open(self.sensTimeSeriesFile, 'a') as f:
                    np.savetxt(f, sensTimeSeries)
                with open(self.energyTimeSeriesFile, 'a') as f:
                    np.savetxt(f, energyTimeSeries)
            sensTimeSeries = []
            energyTimeSeries = []

        for index in range(0, nPerturb):
            writeResult('adjoint', result[index], '{} {}'.format(index, self.scaling))
        self.removeStatusFile()
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readFields', action='store_true')
    parser.add_argument('--matop', action='store_true')
    user, args = parser.parse_known_args()

    adjoint = Adjoint(primal)
    adjoint.initPrimalData()
    adjoint.matop = user.matop
    if not adjoint.matop:
        global matop_petsc
        from adFVM import matop_petsc

    primal.readFields(adjoint.timeSteps[nSteps-writeInterval][0])
    adjoint.compile()

    adjoint.run(user.readFields)

if __name__ == '__main__':
    main()

