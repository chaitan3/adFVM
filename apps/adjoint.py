#!/usr/bin/python2 -u
from __future__ import print_function

from adFVM import config, parallel
from adFVM.parallel import pprint
from adFVM.field import IOField, Field
from adFVM import interp
from adFVM.memory import printMemUsage
from adFVM.postpro import getAdjointViscosity, getAdjointEnergy, getSymmetrizedAdjointEnergy, computeSymmetrizedAdjointEnergy, computeAdjointViscosity, viscositySolver
from adFVM.solver import Solver
from adpy.variable import Variable, Function, Zeros
from adFVM.mesh import cmesh, Mesh

from problem import primal, nSteps, writeInterval, sampleInterval, reportInterval, viscousInterval, perturb, writeResult, nPerturb, parameters, source, adjParams, avgStart, runCheckpoints, startTime

import numpy as np
import time
import os
import argparse

#matop_python = True
matop_python = False
write_M_2norm = False
if matop_python:
    from adFVM import matop_petsc

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
        self.extraArgs = []
        return

    def initFields(self, fields):
        mesh = self.mesh

        for phi, phiN in zip(self.fields, fields):
            phi.field = phiN.field
            phi.defaultComplete()
        return self.fields
        #newFields = self.mapBoundary(*[phi.field for phi in fields] + mesh.getTensor() + mesh.getScalar() + self.getBoundaryTensor(1))
        #return self.getFields(newFields, IOField, refFields=fields)

    def createFields(self):
        fields = []
        for name, dims in zip(self.names, self.dimensions):
            phi = np.zeros((self.mesh.nInternalCells, dims[0]), config.precision)
            fields.append(IOField(name, phi, dims, self.mesh.defaultBoundary))
        self.fields = fields
        for phi in self.fields:
            phi.completeField()
        return fields

    def compileInit(self):
        primal.compileInit()
        #mesh = self.mesh.symMesh
        #meshArgs = mesh.getTensor() + mesh.getScalar()
        #BCArgs = self.getBoundaryTensor(0)
        ## init function
        #rhoa, rhoUa, rhoEa = Variable((mesh.nInternalCells, 1)), Variable((mesh.nInternalCells, 3)), Variable((mesh.nInternalCells, 1)),
        #outputs = Zeros((mesh.nCells, 1)), Zeros((mesh.nCells, 3)), Zeros((mesh.nCells, 1))
        #outputs = self.boundaryInit(*outputs)
        #outputs = self.boundary(*outputs)
        #outputs = self.boundaryEnd(*outputs)
        #rhoaN, rhoUaN, rhoEaN = outputs
        #self.mapBoundary = Function('adjoint_init', [rhoa, rhoUa, rhoEa] + meshArgs + BCArgs, [rhoaN, rhoUaN, rhoEaN])
        return

    def compileSolver(self):
        primal.compileSolver()

        mesh = primal.mesh
        n = len(self.fields)
        scaling = Variable((1, 1))
        # inputs of primal: fields, dt, mesh, source, BC, extra
        gradOutputs, gradInputs = primal.map.grad()
        fields = gradInputs[:n]

        param = parameters[0]
        if param == 'source':
            start = n + 1 + len(mesh.getTensor()) + len(mesh.getScalar())
            end = start + n
        elif param == 'mesh':
            start = n + 1
            end = start + len(Mesh.gradFields)
        elif isinstance(param, tuple):
            assert param[0] == 'BCs'
            _, phi, patchID, key = param
            patch = getattr(primal, phi).phi.BC[patchID]
            index = patch.keys.index(key)
            variable = patch.inputs[index][0]
            start = primal.map._inputs.index(variable)
            end = start + 1
        else:
            raise NotImplementedError
        paramGradient = gradInputs[start:end]

        args = list(primal.map._inputs) + gradOutputs + [scaling]
        outputs = list(fields) + paramGradient
        self.map = Function('primal_grad', args, outputs)
        if self.viscosityType and not matop_python:
            primalFields = primal.map._inputs[:n]
            M_2norm, DT = getAdjointViscosityCpp(*([primal, self.viscosityType] + primalFields + [scaling]))
            outputs = list(viscositySolver(*([primal] + fields + [DT]))) + paramGradient
            if write_M_2norm:
                outputs = outputs + [M_2norm]
            self.viscousMap = Function('primal_grad_viscous', args, outputs)

        fields = [Variable(out.shape, out.dtype) for out in fields]
        primalFields = [Variable(out.shape, out.dtype) for out in fields]
        meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
        args = fields + primalFields
        outputs = computeSymmetrizedAdjointEnergy(primal, *args)
        self.computeEnergy = Function('compute_energy', args + meshArgs, outputs)

            #exit(0)
        #self.map self.map.getAdjoint()

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
            for phi in fields:
                phi.field *= mesh.volumes
        else:
            fields = self.fields
        pprint('STARTING ADJOINT')
        pprint('Number of steps:', nSteps)
        pprint('Write interval:', writeInterval)
        pprint()


        perturbations = []
        for index in range(0, len(perturb)):
            perturbation = perturb[index](None, mesh, 0)
            if isinstance(perturbation, tuple):
                perturbation = list(perturbation)
            if not isinstance(perturbation, list):# or (len(parameters) == 1 and len(perturbation) > 1):
                perturbation = [perturbation]
                # complex parameter perturbation not supported
            perturbations.append(perturbation)


        totalCheckpoints = nSteps//writeInterval
        nCheckpoints = min(firstCheckpoint + runCheckpoints, totalCheckpoints)
        for checkpoint in range(firstCheckpoint, nCheckpoints):
            pprint('PRIMAL FORWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
            primalIndex = nSteps - (checkpoint + 1)*writeInterval
            t = timeSteps[primalIndex, 0]
            dts = timeSteps[primalIndex:primalIndex+writeInterval+1, 1]

            solutions = primal.run(startTime=t, dt=dts, nSteps=writeInterval, mode='forward', reportInterval=reportInterval)

            pprint('ADJOINT BACKWARD RUN {0}/{1}: {2} Steps\n'.format(checkpoint, totalCheckpoints, writeInterval))
            pprint('Time marching for', ' '.join(self.names))

            if checkpoint == 0:
                t, _ = timeSteps[-1]
                if primal.dynamicMesh:
                    lastMesh, lastSolution = solutions[-1]
                    mesh.boundary = lastMesh.boundarydata[m:].reshape(-1,1)
                else:
                    lastSolution = solutions[-1]

                fieldsCopy = [phi.copy() for phi in fields]
                for phi in fields:
                    phi.field /= mesh.volumes
                self.writeFields(fields, t, skipProcessor=True)
                fields = fieldsCopy

            primal.updateSource(source(solutions[-1], mesh, 0))

            pprint('Time step', writeInterval)
            for phi in fields:
                phi.info()
            #energyTimeSeries.append(getAdjointEnergy(primal, *fields))
            inputs = [phi.field for phi in fields + solutions[-1]] + mesh.getTensor() + mesh.getScalar()
            energyTimeSeries.append(self.computeEnergy(*inputs))
            pprint()

            for step in range(0, writeInterval):
                report = ((step + 1) % reportInterval) == 0
                sample = ((step + 1) % sampleInterval) == 0
                viscous = ((step + 1) % viscousInterval) == 0
                
                adjointIndex = writeInterval-1 - step
                t, dt = timeSteps[primalIndex + adjointIndex]
                if primal.dynamicMesh:
                    previousMesh, previousSolution = solutions[adjointIndex]
                    # new mesh boundary
                    mesh.boundary = previousMesh.boundary
                else:
                    previousSolution = solutions[adjointIndex]

                pprint('Time step', adjointIndex)
                if report:
                    pprint('Time marching for', ' '.join(self.names))
                    start = time.time()
                #for index in range(0, n):
                #    fields[index].field *= mesh.volumes

                dtca = np.zeros((1, 1)).astype(config.precision)
                obja = np.ones((1, 1)).astype(config.precision)
                inputs = [phi.field for phi in previousSolution] + \
                     [np.array([[dt]], config.precision)] + \
                     mesh.getTensor() + mesh.getScalar() + \
                     [x[1] for x in primal.sourceTerms] + \
                     primal.getBoundaryTensor(1) + \
                     [x[1] for x in primal.extraArgs] + \
                     [phi.field for phi in fields] + \
                     [dtca, obja] + \
                     [np.array([[self.scaling]], config.precision)]
                options = {'return_static': sample,
                           'zero_static': sample,
                           'return_reusable': report,
                           'replace_reusable': False,
                           }

                start10 = time.time()
                if self.viscosityType and not matop_python and viscous:
                    outputs = self.viscousMap(*inputs, **options)
                else:
                    outputs = self.map(*inputs, **options)
                pprint(time.time()-start10)

                #print(sum([(1e-3*phi).sum() for phi in gradient]))
                #inp1 = inputs[:3] + inputs[-3:-1]
                #inp2 = [phi + 1e-3 for phi in inputs[:3]] + inputs[-3:-1]
                #x1 = primal.map(*inp1)[-2]
                #x2 = primal.map(*inp2)[-2]
                #print(x1, x2, x1-x2)
                #import pdb;pdb.set_trace()

                # gradients
                n = len(fields)
                gradient = outputs[:n]
                for index in range(0, n):
                    fields[index].field = gradient[index]
                    #fields[index].field = gradient[index]/mesh.volumes

                if matop_python:
                    stackedFields = np.concatenate([phi.field/mesh.volumes for phi in fields], axis=1)
                    stackedFields = np.ascontiguousarray(stackedFields)

                    inputs = previousSolution + [self.scaling]
                    kwargs = {'visc': self.viscosityType, 'scale': self.viscosityScaler, 'report':report}
                    weight = interp.centralOld(getAdjointViscosity(*inputs, **kwargs), mesh)
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
                        phi.field = np.ascontiguousarray(phi.field)*mesh.volumes

                #print([type(x) for x in outputs])
                if sample:
                    paramGradient = list(outputs[n:])

                    # compute sensitivity using adjoint solution
                    sensitivities = []
                    for index, perturbation in enumerate(perturbations):
                        # make efficient cpu implementation
                        #sensitivity = 0.
                        #for derivative, delphi in zip(paramGradient, perturbation):
                        #    sensitivity += np.sum(derivative * delphi)
                        sensitivity = cmesh.computeSensitivity(paramGradient, perturbation)
                        sensitivities.append(sensitivity)
                    sensitivities = parallel.sum(sensitivities, allreduce=False)
                    if (nSteps - (primalIndex + adjointIndex)) > avgStart:
                        for index in range(0, len(perturb)):
                            result[index] += sensitivities[index]
                    sensitivities = [sens/sampleInterval for sens in sensitivities]
                    for i in range(0, sampleInterval):
                        sensTimeSeries.append(sensitivities)

                if report:
                    for phi in fields:
                        phi.info()
                    #energyTimeSeries.append(getAdjointEnergy(primal, *fields))
                    inputs = [phi.field for phi in fields + previousSolution] + mesh.getTensor() + mesh.getScalar()
                    energyTimeSeries.append(self.computeEnergy(*inputs))
                    end = time.time()
                    pprint('Time for adjoint iteration: {0}'.format(end-start))
                    pprint('Time since beginning:', end-config.runtime)
                    pprint('Simulation Time and step: {0}, {1}'.format(*timeSteps[primalIndex + adjointIndex + 1]))
                pprint()
                #parallel.mpi.Barrier()

            #exit(1)
            #print(fields[0].field.max())
            fieldsCopy = [phi.copy() for phi in fields]
            for phi in fields:
                phi.field /= mesh.volumes
            self.writeFields(fields, t, skipProcessor=True)
            fields = fieldsCopy

            # write M_2norm
            if write_M_2norm:
                with IOField.handle(t):
                    IOField('M_2norm', outputs[-1], (1,)).write()

            #for phi in fields:
            #    phi.field /= mesh.volumes
            #self.writeFields(fields, t, skipProcessor=True)
            #for phi in fields:
            #    phi.field *= mesh.volumes

            #print(fields[0].field.max())
            self.writeStatusFile([checkpoint + 1, result])
            #energyTimeSeries = mpi.gather(timeSeries, root=0)
            if parallel.rank == 0:
                with open(self.sensTimeSeriesFile, 'ab') as f:
                    np.savetxt(f, sensTimeSeries)
                with open(self.energyTimeSeriesFile, 'ab') as f:
                    np.savetxt(f, energyTimeSeries)
            sensTimeSeries = []
            energyTimeSeries = []
        #pprint(checkpoint, totalCheckpoints)

        if checkpoint + 1 == totalCheckpoints:
            writeResult('adjoint', result, str(self.scaling), self.sensTimeSeriesFile)
            #for index in range(0, nPerturb):
            #    writeResult('adjoint', result[index], '{} {}'.format(index, self.scaling))
            self.removeStatusFile()
        return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--readFields', action='store_true')
    user, args = parser.parse_known_args()

    adjoint = Adjoint(primal)

    primal.readFields(startTime)
    adjoint.createFields()
    adjoint.compile()

    adjoint.initPrimalData()

    adjoint.run(user.readFields)

if __name__ == '__main__':
    main()

