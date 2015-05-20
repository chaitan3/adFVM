import numpy as np
import time
import cPickle as pickle
import os

import config, parallel
from config import ad, T
from parallel import pprint

from field import Field, CellField, IOField
from mesh import Mesh
import timestep

logger = config.Logger(__name__)

class Solver(object):
    defaultConfig = {
                        'timeIntegrator': 'euler',
                        'source': None,
                        'adjoint': False
                    }

    def __init__(self, case, **userConfig):
        logger.info('initializing solver for {0}'.format(case))
        fullConfig = self.__class__.defaultConfig
        fullConfig.update(userConfig)
        for key in fullConfig:
            setattr(self, key, fullConfig[key])

        self.mesh = Mesh.create(case)
        Field.setSolver(self)

        self.timeIntegrator = getattr(timestep, self.timeIntegrator)
        self.padField = PadFieldOp(self.mesh)
        self.gradPadField = gradPadFieldOp(self.mesh)

    def compile(self):
        pprint('Compiling solver', self.__class__.defaultConfig['timeIntegrator'])

        self.dt = ad.scalar()

        #paddedStackedFields = ad.matrix()
        #paddedStackedFields.tag.test_value = np.random.rand(self.mesh.paddedMesh.origMesh.nCells, 5).astype(config.precision)
        #fields = self.unstackFields(paddedStackedFields, CellField)
        #fields = self.timeIntegrator(self.equation, self.boundary, fields, self)
        #newStackedFields = self.stackFields(fields, ad)
        #self.forward = T.function([paddedStackedFields], [newStackedFields, self.dtc, self.local, self.remote], on_unused_input='warn')#, mode=T.compile.MonitorMode(pre_func=config.inspect_inputs, post_func=config.inspect_outputs))

        stackedFields = ad.matrix()
        newStackedFields = self.timeIntegrator(self.equation, self.boundary, stackedFields, self)
        self.forward = self.function([stackedFields, self.dt], \
                       [newStackedFields, self.dtc, self.local, self.remote], 'forward')
        pprint()
        if self.adjoint:
            stackedAdjointFields = ad.matrix()
            #paddedGradient = ad.grad(ad.sum(newStackedFields*stackedAdjointFields), paddedStackedFields)
            #self.gradient = T.function([paddedStackedFields, stackedAdjointFields], paddedGradient)
            gradient = ad.grad(ad.sum(newStackedFields*stackedAdjointFields), stackedFields)
            self.gradient = self.function([stackedFields, stackedAdjointFields, self.dt], \
                            gradient, 'gradient')

    def stackFields(self, fields, mod): 
        return mod.concatenate([phi.field for phi in fields], axis=1)

    def unstackFields(self, stackedFields, mod, names=None):
        if names is None:
            names = self.names
        fields = []
        nDimensions = np.concatenate(([0], np.cumsum(np.array(self.dimensions))))
        nDimensions = zip(nDimensions[:-1], nDimensions[1:])
        for name, dim, dimRange in zip(names, self.dimensions, nDimensions):
            phi = stackedFields[:, range(*dimRange)]
            fields.append(mod(name, phi, dim))
        return fields


    def run(self, endTime=np.inf, writeInterval=config.LARGE, startTime=0.0, dt=1e-3, nSteps=config.LARGE, 
            mode='simulation', objective=lambda x: 0, perturb=None):

        logger.info('running solver for {0}'.format(nSteps))
        mesh = self.mesh
        #initialize
        fields = self.initFields(startTime)
        pprint()

        if not hasattr(self, 'forward'):
            self.compile()

        t = startTime
        dts = dt
        timeIndex = 0
        if isinstance(dts, np.ndarray):
            dt = dts[timeIndex]
        stackedFields = self.stackFields(fields, np)
        
        timeSteps = []
        # objective is local
        if perturb is not None:
            perturb(stackedFields, t)
        result = objective(stackedFields)
        # writing and returning local solutions
        if mode == 'forward':
            solutions = [stackedFields]


        while t < endTime and timeIndex < nSteps:
            #import resource; print resource.getrusage(resource.RUSAGE_SELF)[2]*resource.getpagesize()/(1024*1024)
            #import guppy; print guppy.hpy().heap()
            start = time.time()

            pprint('Time marching for', ' '.join(self.names))
            for index in range(0, len(fields)):
                fields[index].info()

            # mpi stuff, bloat stackedFields
            #TESTING
            #stackedFields = parallel.getRemoteCells(stackedFields, mesh)  

            pprint('Time step', timeIndex)
            #stackedFields, dtc = self.forward(stackedFields)
            stackedFields, dtc, local, remote = self.forward(stackedFields, dt)
            #print local.shape, local.dtype, local, np.abs(local).max(), np.abs(local).min()
            #print remote.shape, remote.dtype, remote, np.abs(remote).max(), np.abs(remote).min()
            #print local
            #print remote

            #lStart = 0
            #rStart = 0
            #print parallel.rank, mesh.remotePatches
            #for patchID in mesh.remotePatches:
            #    n = mesh.boundary[patchID]['nFaces']
            #    #n = len(mesh.paddedMesh.localRemoteFaces[self.loc][patchID])
            #    np.savetxt('local_' + patchID, local[lStart:lStart+n])
            #    #print 'local', patchID, local[lStart:lStart+n], local.shape
            #    lStart += n
            #    #n = mesh.paddedMesh.remoteFaces[self.loc][patchID]
            #    np.savetxt('remote_' + patchID, remote[rStart:rStart+n])
            #    #print 'remote', patchID, remote[rStart:rStart+n], remote.shape
            #    rStart += n


            fields = self.unstackFields(stackedFields, IOField)
            # TODO: fix unstacking F_CONTIGUOUS
            for phi in fields:
                phi.field = np.ascontiguousarray(phi.field)

            end = time.time()
            pprint('Time for iteration:', end-start)
            pprint('objective: ', parallel.sum(result))
            
            result += objective(stackedFields)
            timeSteps.append([t, dt])
            if mode == 'forward':
                solutions.append(stackedFields)

            t = round(t+dt, 9)
            timeIndex += 1
            pprint('Simulation Time:', t, 'Time step:', dt)
            if timeIndex % writeInterval == 0:
                self.writeFields(fields, t)
            pprint()

            # compute dt for next time step
            dt = min(parallel.min(dtc), dt*self.stepFactor, endTime-t)
            if isinstance(dts, np.ndarray):
                dt = dts[timeIndex]

        if mode == 'forward':
            return solutions
        if (timeIndex % writeInterval != 0) and (timeIndex >= writeInterval):
            self.writeFields(fields, t)
        return timeSteps, result

    def function(self, inputs, outputs, name):
        return SolverFunction(inputs, outputs, self, name)

class SolverFunction(object):
    counter = 0
    def __init__(self, inputs, outputs, solver, name):
        logger.info('compiling function')
        self.symbolic = []
        self.values = []
        mesh = solver.mesh
        self.populate_mesh(self.symbolic, mesh, mesh.paddedMesh, mesh.origPatches)
        self.populate_mesh(self.values, mesh.origMesh, mesh.paddedMesh.origMesh, mesh.origPatches)
        self.populate_BCs(self.symbolic, solver, 0)
        self.populate_BCs(self.values, solver, 1)
        self.generate(inputs, outputs, solver.mesh.case, name)

    def populate_mesh(self, inputs, mesh, paddedMesh, origPatches):
        attrs = Mesh.fields + Mesh.constants
        for attr in attrs:
            if attr == 'boundary':
                for patchID in origPatches:
                    patch = getattr(mesh, attr)[patchID]
                    inputs.append(patch['startFace'])
                    inputs.append(patch['nFaces'])
            else:
                inputs.append(getattr(mesh, attr))
                if parallel.nProcessors > 1:
                    inputs.append(getattr(paddedMesh, attr))

    def populate_BCs(self, inputs, solver, index):
        fields = solver.getBCFields()
        for phi in fields:
            if hasattr(phi, 'BC'):
                for patchID in phi.BC:
                    inputs.extend([value[index] for value in phi.BC[patchID].inputs])

    def generate(self, inputs, outputs, caseDir, name):
        SolverFunction.counter += 1
        pklFile = caseDir + 'func_{0}.pkl'.format(name)
        inputs.extend(self.symbolic)

        fn = None
        if parallel.rank == 0:
            start = time.time()
            if os.path.exists(pklFile) and config.allowUnpicklingFunction:
                pprint('Loading pickled file', pklFile)
                pkl = open(pklFile).read()
            else:
                fn = T.function(inputs, outputs, on_unused_input='ignore', mode=config.compile_mode)
                #T.printing.pydotprint(fn, outfile='graph.png')
                pkl = pickle.dumps(fn)
                pprint('Saving pickle file', pklFile)
                f = open(pklFile, 'w').write(pkl)
            end = time.time()
            pprint('Module size: {0:.2f}'.format(float(len(pkl))/(1024*1024)))
            pprint('Compilation time: {0:.2f}'.format(end-start))
        else:
            pkl = None

        if parallel.nProcessors > 1:
            start = time.time()
            pkl = parallel.mpi.bcast(pkl, root=0)
            parallel.mpi.Barrier()
            end = time.time()
            pprint('Transfer time: {0:.2f}'.format(end-start))

        start = time.time()
        if fn is None:
            fn = pickle.loads(pkl)
        parallel.mpi.Barrier()
        end = time.time()
        pprint('Loading time: {0:.2f}'.format(end-start))

        self.fn = fn

    def __call__(self, *inputs):
        logger.info('running function')
        inputs = list(inputs)
        inputs.extend(self.values)
        return self.fn(*inputs)


class PadFieldOp(T.Op):
    __props__ = ()

    def __init__(self, mesh):
        #self.mesh = mesh
        pass

    def make_node(self, x):
        assert hasattr(self, '_props')
        x = ad.as_tensor_variable(x)
        return T.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        #assert inputs[0].flags['C_CONTIGUOUS'] == True
        #output_storage[0][0] = parallel.getRemoteCells(inputs[0], self.mesh)
        output_storage[0][0] = parallel.getRemoteCells(np.ascontiguousarray(inputs[0]), Field.mesh)

    def grad(self, inputs, output_grads):
        return [self.solver.gradPadField(output_grads[0])]

class gradPadFieldOp(T.Op):
    __props__ = ()

    def __init__(self, mesh):
        #self.mesh = mesh
        pass

    def make_node(self, x):
        assert hasattr(self, '_props')
        x = ad.as_tensor_variable(x)
        return T.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        #assert inputs[0].flags['C_CONTIGUOUS'] == True
        #output_storage[0][0] = parallel.getAdjointRemoteCells(inputs[0], self.mesh)
        output_storage[0][0] = parallel.getAdjointRemoteCells(np.ascontiguousarray(inputs[0]), Field.mesh)
