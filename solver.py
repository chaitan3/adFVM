import numpy as np
import time
import cPickle as pickle
import os

import config, parallel
from config import ad, T
from parallel import pprint

from field import Field, CellField, IOField
from mesh import Mesh

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

        self.timeIntegrator = globals()[self.timeIntegrator]
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
        self.forward = self.function([stackedFields, self.dt], [newStackedFields, self.dtc, self.local, self.remote])
        pprint()
        if self.adjoint:
            stackedAdjointFields = ad.matrix()
            #paddedGradient = ad.grad(ad.sum(newStackedFields*stackedAdjointFields), paddedStackedFields)
            #self.gradient = T.function([paddedStackedFields, stackedAdjointFields], paddedGradient)
            gradient = ad.grad(ad.sum(newStackedFields*stackedAdjointFields), stackedFields)
            self.gradient = self.function([stackedFields, stackedAdjointFields, self.dt], gradient)

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

    def function(self, inputs, outputs):
        return SolverFunction(inputs, outputs, self)

class SolverFunction(object):
    counter = 0
    def __init__(self, inputs, outputs, solver):
        logger.info('compiling function')
        self.symbolic = []
        self.values = []
        mesh = solver.mesh
        self.populate_mesh(self.symbolic, mesh, mesh.paddedMesh, mesh.origPatches)
        self.populate_mesh(self.values, mesh.origMesh, mesh.paddedMesh.origMesh, mesh.origPatches)
        self.populate_BCs(self.symbolic, solver, 0)
        self.populate_BCs(self.values, solver, 1)
        self.generate(inputs, outputs, solver.mesh.case)

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
            if hasattr(phi, 'phi'):
                for patchID in phi.phi.BC:
                    inputs.extend([value[index] for value in phi.phi.BC[patchID].inputs])

    def generate(self, inputs, outputs, caseDir):
        SolverFunction.counter += 1
        pklFile = caseDir + 'func_{0}.pkl'.format(SolverFunction.counter)
        inputs.extend(self.symbolic)

        if parallel.rank == 0:
            start = time.time()
            if os.path.exists(pklFile):
                pkl = open(pklFile).read()
                fn = pickle.loads(pkl)
                pprint('Loading pickled file', pklFile)
            else:
                fn = T.function(inputs, outputs, on_unused_input='ignore', mode=config.compile_mode)
                #T.printing.pydotprint(fn, outfile='graph.png')
                pkl = pickle.dumps(fn)
                f = open(pklFile, 'w')
                f.write(pkl)
                f.close()
                pprint('Saving pickle file', pklFile)
            end = time.time()
            pprint('Compilation time: {0:.2f}'.format(end-start))
            pprint('Compilation size: {0:.2f}'.format(float(len(pkl))/(1024*1024)))
        else:
            fn = None

        if parallel.nProcessors > 1:
            start = time.time()
            fn = parallel.mpi.bcast(fn, root=0)
            parallel.mpi.Barrier()
            end = time.time()
            pprint('Transfer time: {0:.2f}'.format(end-start))

        self.fn = fn

    def __call__(self, *inputs):
        logger.info('running function')
        inputs = list(inputs)
        inputs.extend(self.values)
        return self.fn(*inputs)

def euler(equation, boundary, stackedFields, solver):
    paddedStackedFields = solver.padField(stackedFields)
    paddedFields = solver.unstackFields(paddedStackedFields, CellField)
    LHS = equation(*paddedFields)
    internalFields = [(paddedFields[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(paddedFields))]
    newFields = boundary(*internalFields)
    return solver.stackFields(newFields, ad)

# classical
def RK2(equation, boundary, stackedFields, solver):
    paddedStackedFields0 = solver.padField(stackedFields)
    paddedFields0 = solver.unstackFields(paddedStackedFields0, CellField)
    LHS = equation(*paddedFields0)
    internalFields = [(paddedFields0[index].getInternalField() - LHS[index].field*solver.dt/2) for index in range(0, len(paddedFields0))]
    fields1 = boundary(*internalFields)

    paddedStackedFields1 = solver.padField(solver.stackFields(fields1, ad))
    paddedFields1 = solver.unstackFields(paddedStackedFields1, CellField)
    LHS = equation(*paddedFields1)
    internalFields = [(paddedFields0[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(paddedFields0))]

    newFields = boundary(*internalFields)
    return solver.stackFields(newFields, ad)

# classical
def RK4(equation, boundary, stackedFields, solver):
    fields = solver.unstackFields(stackedFields, CellField)
    def NewFields(a, LHS):
        internalFields = [phi.getInternalField() for phi in fields]
        for termIndex in range(0, len(a)):
            for index in range(0, len(fields)):
                internalFields[index] -= a[termIndex]*LHS[termIndex][index].field*solver.dt
        return boundary(*internalFields)

    def f(a, *LHS):
        if len(LHS) != 0:
            newFields = NewFields(a, LHS)
        else:
            newFields = fields
        paddedStackedFields = solver.padField(solver.stackFields(newFields, ad))
        paddedFields = solver.unstackFields(paddedStackedFields, CellField)
        return equation(*paddedFields)

    k1 = f([0.])
    k2 = f([0.5], k1)
    k3 = f([0.5], k2)
    k4 = f([1.], k3)
    newFields = NewFields([1./6, 1./3, 1./3, 1./6], [k1, k2, k3, k4])

    return solver.stackFields(newFields, ad)

# classical
def SSPRK(equation, boundary, stackedFields, solver):
    # 2nd order
    #alpha = np.array([[1.,0],[1./2,1./2]])
    #beta = np.array([[1,0],[0,1./2]])
    # 3rd order
    alpha = np.array([[1,0,0],[3./4,1./4, 0],[1./3,0,2./3]])
    beta = np.array([[1,0,0],[0,1./4,0],[0,0,2./3]])
    nStages = alpha.shape[0]
    LHS = []
    fields = []
    fields.append(solver.unstackFields(stackedFields, CellField))
    nFields = len(fields[0])
    for i in range(0, nStages):
        paddedStackedFields = solver.padField(solver.stackFields(fields[i], ad))
        paddedFields = solver.unstackFields(paddedStackedFields, CellField)
        LHS.append(equation(*paddedFields))
        internalFields = [0]*nFields
        for j in range(0, i+1):
            for index in range(0, nFields):
                internalFields[index] += alpha[i,j]*fields[j][index].getInternalField()-beta[i,j]*LHS[j][index].field*solver.dt
        fields.append(boundary(*internalFields))
        
    return solver.stackFields(fields[-1], ad)

# DOES NOT WORK
def implicit(equation, boundary, fields, garbage):
    assert ad.__name__ == 'numpad'
    start = time.time()

    names = [phi.name for phi in fields]
    pprint('Solving for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = CellField.copy(fields[index])
        fields[index].info()
    nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in fields]))))
    nDimensions = zip(nDimensions[:-1], nDimensions[1:])
    def setInternalFields(stackedInternalFields):
        internalFields = []
        # range creates a copy on the array
        for index in range(0, len(fields)):
            internalFields.append(stackedInternalFields[:, range(*nDimensions[index])])
        return boundary(*internalFields)
    def solver(internalFields):
        newFields = setInternalFields(internalFields)
        for index in range(0, len(fields)):
            newFields[index].old = fields[index].old
        return ad.hstack([phi.field for phi in equation(*newFields)])

    internalFields = ad.hstack([phi.getInternalField() for phi in fields])
    solution = ad.solve(solver, internalFields)
    newFields = setInternalFields(solution)
    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name

    end = time.time()
    pprint('Time for iteration:', end-start)
    return newFields

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
