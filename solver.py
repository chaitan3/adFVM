import numpy as np
import time
import cPickle as pkl
import os
import copy

import config, parallel
from config import ad, T
from parallel import pprint
from compat import printMemUsage

from field import Field, CellField, IOField
from mesh import Mesh
import timestep

logger = config.Logger(__name__)

class Solver(object):
    defaultConfig = {
                        'timeIntegrator': 'euler',
                        'objective': lambda x: 0,
                        'adjoint': False,
                        'dynamicMesh': False,
                        'sourceTerm': None,
                        'localTimeStep': False,
                        'postpro': []
                    }

    def __init__(self, case, **userConfig):
        logger.info('initializing solver for {0}'.format(case))
        fullConfig = self.__class__.defaultConfig
        fullConfig.update(userConfig)
        for key in fullConfig:
            setattr(self, key, fullConfig[key])

        self.mesh = Mesh.create(case)
        self.resultFile = self.mesh.case + 'objective.txt'
        self.statusFile = self.mesh.case + 'status.pkl'
        self.timeSeriesFile = self.mesh.case + 'timeSeries.txt'
        Field.setSolver(self)

        self.timeStepCoeff = getattr(timestep, self.timeIntegrator)()
        self.nStages = self.timeStepCoeff[0].shape[0]
        self.stage = 0
        self.init = None

    def compile(self):
        pprint('Compiling solver', self.__class__.defaultConfig['timeIntegrator'])
        if self.localTimeStep:
            self.dt = ad.bcmatrix()
        else:
            self.dt = ad.scalar()
        self.t0 = ad.scalar()
        self.t = self.t0*1.
        stackedFields = ad.matrix()
        newStackedFields = timestep.timeStepper(self.equation, self.boundary, stackedFields, self)
        self.forward = self.function([stackedFields, self.dt, self.t0], \
                       [newStackedFields, self.dtc, self.local, self.remote], 'forward')
        if self.adjoint:
            stackedAdjointFields = ad.matrix()
            scalarFields = ad.sum(newStackedFields*stackedAdjointFields)
            gradientInputs = [stackedFields] + self.sourceVariables
            gradients = ad.grad(scalarFields, gradientInputs)
            #meshGradient = ad.grad(scalarFields, mesh)
            self.gradient = self.function([stackedFields, stackedAdjointFields, self.dt, self.t], \
                            gradients, 'adjoint')
            #self.tangent = self.function([stackedFields, stackedAdjointFields, self.dt], \
            #                ad.Rop(newStackedFields, stackedFields, stackedAdjointFields), 'tangent')
        if config.compile:
            exit()
        pprint()

    def stackFields(self, fields, mod): 
        return mod.concatenate([phi.field for phi in fields], axis=1)

    def unstackFields(self, stackedFields, mod, names=None, **kwargs):
        if names is None:
            names = self.names
        fields = []
        nDimensions = np.concatenate(([0], np.cumsum(np.array(self.dimensions))))
        nDimensions = zip(nDimensions[:-1], nDimensions[1:])
        for name, dim, dimRange in zip(names, self.dimensions, nDimensions):
            phi = stackedFields[:, range(*dimRange)]
            fields.append(mod(name, phi, dim, **kwargs))
        return fields

    def run(self, endTime=np.inf, writeInterval=config.LARGE, startTime=0.0, dt=1e-3, nSteps=config.LARGE, \
            startIndex=0, result=0., mode='simulation'):

        logger.info('running solver for {0}'.format(nSteps))
        mesh = self.mesh
        #initialize
        if self.dynamicMesh:
            mesh.read(startTime)
        fields = self.initFields(startTime)
        pprint()

        t = startTime
        dts = dt
        timeIndex = startIndex
        lastIndex = 0
        if self.localTimeStep:
            dt = dts*np.ones_like(mesh.origMesh.volumes)
        elif isinstance(dts, np.ndarray):
            dt = dts[timeIndex]
        stackedFields = self.stackFields(fields, np)

        # objective is local
        result += self.objective(stackedFields)
        timeSteps = []
        timeSeries = []
        # writing and returning local solutions
        if mode == 'forward':
            if self.dynamicMesh:
                instMesh = Mesh()
                instMesh.boundary = copy.deepcopy(self.mesh.origMesh.boundary)
                solutions = [(instMesh, stackedFields)]
            else:
                solutions = [stackedFields]

        pprint('Time marching for', ' '.join(self.names))

        while t < endTime and timeIndex < nSteps:
            printMemUsage()
            start = time.time()

            for index in range(0, len(fields)):
                fields[index].info()

            pprint('Time step', timeIndex)
            #stackedFields, dtc = self.forward(stackedFields)
            stackedFields, dtc, local, remote = self.forward(stackedFields, dt, t)
            #print local.shape, local.dtype, (local).max(), (local).min(), np.isnan(local).any()
            #print remote.shape, remote.dtype, np.abs(remote).max(), np.abs(remote).min(), (remote).max(), (remote).min(), np.isnan(remote).any()

            fields = self.unstackFields(stackedFields, IOField)
            #from plot.visualize import plot
            #Z = plot(fields[1], 'patch1_half0', 'scripts/images/fourier_0000.png')
            #from scripts.fft import fft
            #fft(Z, 'scripts/images/fourier_fft.png')
            #exit(1)

            # TODO: fix unstacking F_CONTIGUOUS
            for phi in fields:
                phi.field = np.ascontiguousarray(phi.field)

            parallel.mpi.Barrier()
            end = time.time()
            pprint('Time for iteration:', end-start)
            pprint('Time since beginning:', end-config.runtime)
            pprint('cumulative objective: ', parallel.sum(result))

            mesh.update(t, dt)
            
            timeSteps.append([t, dt])
            instObjective = self.objective(stackedFields)
            result += instObjective
            timeSeries.append(parallel.sum(instObjective)/(nSteps + 1))
            if mode == 'forward':
                if self.dynamicMesh:
                    instMesh = Mesh()
                    instMesh.boundary = copy.deepcopy(self.mesh.origMesh.boundary)
                    solutions.append((instMesh, stackedFields))
                else:
                    solutions.append(stackedFields)
            timeIndex += 1
            if self.localTimeStep:
                pprint('Simulation Time:', t, 'Time step: min', parallel.min(dt), 'max', parallel.max(dt))
                t += 1
            else:
                pprint('Simulation Time:', t, 'Time step:', dt)
                t = round(t+dt, 9)
            # compute dt for next time step
            if self.localTimeStep:
                dt = dtc
            elif isinstance(dts, np.ndarray):
                dt = dts[timeIndex]
            else:
                dt = min(parallel.min(dtc), dt*self.stepFactor, endTime-t)

            #local = IOField('local', local, (local.shape[1],))
            #local.partialComplete()
            #local.write(t)

            if (timeIndex % writeInterval == 0) and (mode != 'forward'):
                # write mesh, fields, status
                if self.dynamicMesh:
                    mesh.write(t)
                dtc = IOField('dtc', dtc, (1,))
                dtc.partialComplete()
                self.writeFields(fields + [dtc], t)
                with open(self.statusFile, 'wb') as status:
                    pkl.dump([timeIndex, t, dt, result], status)

                # write timeSeries if in orig mode (problem.py)
                if mode == 'orig' and parallel.rank == 0 and not self.localTimeStep:
                    with open(self.timeStepFile, 'a') as f:
                        np.savetxt(f, timeSteps[lastIndex:])
                    with open(self.timeSeriesFile, 'a') as f:
                        np.savetxt(f, timeSeries[lastIndex:])
                    lastIndex = len(timeSteps)
            pprint()


        if mode == 'forward':
            return solutions
        if (timeIndex % writeInterval != 0) and (timeIndex >= writeInterval):
            dtc = IOField('dtc', dtc, (1,))
            dtc.partialComplete()
            self.writeFields(fields + [dtc], t)
        return result

    def function(self, inputs, outputs, name, **kwargs):
        return SolverFunction(inputs, outputs, self, name, **kwargs)

class SolverFunction(object):
    counter = 0
    def __init__(self, inputs, outputs, solver, name, BCs=True, postpro=False, source=True):
        logger.info('compiling function')
        self.symbolic = []
        self.values = []
        mesh = solver.mesh
        self.populate_mesh(self.symbolic, mesh, mesh)
        self.populate_mesh(self.values, mesh.origMesh, mesh)
        if BCs:
            self.populate_BCs(self.symbolic, solver, 0)
            self.populate_BCs(self.values, solver, 1)
        # source terms
        if source:
            self.symbolic.extend(solver.sourceVariables)
            self.values.extend(solver.sourceTerm)
        # postpro variables
        if postpro and len(solver.postpro) > 0:
            symbolic, values = zip(*solver.postpro)
            self.symbolic.extend(symbolic)
            self.values.extend(values)

        self.generate(inputs, outputs, solver.mesh.case, name)

    def populate_mesh(self, inputs, mesh, solverMesh):
        attrs = Mesh.fields + Mesh.constants
        for attr in attrs:
            inputs.append(getattr(mesh, attr))
        for patchID in solverMesh.origPatches:
            for attr in solverMesh.getBoundaryTensor(patchID):
                inputs.append(mesh.boundary[patchID][attr[0]])

    def populate_BCs(self, inputs, solver, index):
        fields = solver.getBCFields()
        for phi in fields:
            if hasattr(phi, 'BC'):
                for patchID in solver.mesh.localPatches:
                    inputs.extend([value[index] for value in phi.BC[patchID].inputs])

    def generate(self, inputs, outputs, caseDir, name):
        SolverFunction.counter += 1
        pklFile = caseDir + '{0}_func_{1}.pkl'.format(config.device, name)
        inputs.extend(self.symbolic)

        fn = None
        pklData = None
        if parallel.rank == 0:
            start = time.time()
            if os.path.exists(pklFile) and config.unpickleFunction:
                pprint('Loading pickled file', pklFile)
                pklData = open(pklFile).read()
            else:
                fn = T.function(inputs, outputs, on_unused_input='ignore', mode=config.compile_mode)
                #T.printing.pydotprint(fn, outfile='graph.png')
                if config.pickleFunction or (parallel.nProcessors > 1):
                    pklData = pkl.dumps(fn)
                    pprint('Saving pickle file', pklFile)
                    f = open(pklFile, 'w').write(pklData)
                    pprint('Module size: {0:.2f}'.format(float(len(pklData))/(1024*1024)))
            end = time.time()
            pprint('Compilation time: {0:.2f}'.format(end-start))

        if not config.compile:
            start = time.time()
            pklData = parallel.mpi.bcast(pklData, root=0)
            if parallel.mpi.bcast(fn is not None, root=0) and parallel.nProcessors > 1:
                T.gof.cc.get_module_cache().refresh(cleanup=False)
            end = time.time()
            pprint('Transfer time: {0:.2f}'.format(end-start))

            start = time.time()
            unloadingStages = config.user.unloadingStages
            coresPerNode = config.user.coresPerNode
            coresPerStage = coresPerNode/unloadingStages
            nodeStage = (parallel.rank % coresPerNode)/coresPerStage
            for stage in range(unloadingStages):
                printMemUsage()
                if (nodeStage == stage) and (fn is None):
                    fn = pkl.loads(pklData)
                parallel.mpi.Barrier()
            end = time.time()
            pprint('Loading time: {0:.2f}'.format(end-start))
            printMemUsage()

        self.fn = fn

    def __call__(self, *inputs):
        logger.info('running function')
        inputs = list(inputs)
        #print 'get', id(self.values[29].data)
        inputs.extend(self.values)
        return self.fn(*inputs)

