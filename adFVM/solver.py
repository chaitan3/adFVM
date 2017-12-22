import numpy as np
import time
import pickle as pkl
import os
import copy

#import adFVMcpp
from . import config, parallel, timestep
from .parallel import pprint
from .memory import printMemUsage

from .field import Field, CellField, IOField
from .mesh import Mesh
from .mesh import extractField

from adpy.tensor import StaticVariable, ExternalFunctionOp, Function

logger = config.Logger(__name__)

class Solver(object):
    defaultConfig = {
                        'timeIntegrator': 'euler',
                        'objective': None,
                        'objectiveString': None,
                        'adjoint': False,
                        'dynamicMesh': False,
                        'localTimeStep': False,
                        'fixedTimeStep': False,
                        'stepFactor': 1.0,
                        'postpro': [],
                        'sourceTerms': [],
                        'timeSeriesAppend': ''
                    }

    def __init__(self, case, **userConfig):
        logger.info('initializing solver for {0}'.format(case))
        fullConfig = self.__class__.defaultConfig
        for key in userConfig:
            assert key in fullConfig
        fullConfig.update(userConfig)
        for key in fullConfig:
            setattr(self, key, fullConfig[key])

        caser = os.path.realpath(case)
        pprint('Starting', caser)
        self.mesh = Mesh.create(case)
        self.resultFile = self.mesh.case + 'objective.txt'
        self.statusFile = self.mesh.case + 'status.pkl'
        self.timeSeriesFile = self.mesh.case + 'timeSeries{}.txt'.format(self.timeSeriesAppend)
        Field.setSolver(self)
        #Field.setMesh(self.mesh)

        self.timeStepCoeff = getattr(timestep, self.timeIntegrator)()
        self.nStages = self.timeStepCoeff[0].shape[0]
        self.stage = 0
        self.init = None
        self.firstRun = True
        self.extraArgs = []
        return

    def compile(self, adjoint=None):
        pprint('Compiling solver', self.__class__.defaultConfig['timeIntegrator'])
        self.compileInit()
        self.compileSolver()
        Function.createCodeDir(self.mesh.caseDir, replace=config.compile)
        parallel.mpi.Barrier()
        Function.compile(case=None, init=False, replace=config.compile, compiler_args=config.get_compiler_args())
        if config.compile_exit:
            exit(0)
        Function.initialize(parallel.localRank, self.mesh)
        return
        
    def getFields(self, fields, mod, refFields=None):
        cellFields = []
        if refFields is None:
            refFields = self.fields
        for phi, phiO in zip(fields, refFields):
            if isinstance(phi, Field):
                phi = phi.field
            name, dim = phiO.name, phiO.dimensions
            cellFields.append(mod(name, phi, dim))
        return cellFields

    def getBoundaryTensor(self, index=0):
        return sum([phi.getTensor(index) for phi in self.fields], [])

    def boundaryInit(self, *fields):
        mesh = self.mesh.symMesh
        fields = list(fields)
        for index, phi in enumerate(fields):
            (phi,) = ExternalFunctionOp('mpi_init', (phi, mesh.owner), (phi,)).outputs
            fields[index] = phi
        fields = tuple(fields)
        fields = ExternalFunctionOp('mpi_dummy', fields, fields, empty=True).outputs
        return fields

    def boundaryEnd(self, *fields):
        fields = ExternalFunctionOp('mpi_dummy', fields, fields, empty=True).outputs
        fields = list(fields)
        for index, phi in enumerate(fields):
            (phi,) = ExternalFunctionOp('mpi_end', (phi,), (phi,)).outputs
            fields[index] = phi
        fields = tuple(fields)
        return fields

    def boundary(self, *fields, **kwargs):
        boundaryFields = kwargs['boundary']
        fields = list(fields)
        for index, phi in enumerate(fields):
            if boundaryFields is not None:
                phi = boundaryFields[index].updateGhostCells(phi)
            (phi,) = ExternalFunctionOp('mpi', (phi,), (phi,)).outputs
            fields[index] = phi
        return tuple(fields)

    def initSource(self):
        mesh = self.mesh.symMesh
        symbolics = [StaticVariable((mesh.nInternalCells,) + dims) for dims in self.dimensions]
        values = [np.zeros((self.mesh.nInternalCells, dims[0]), config.precision) for dims in self.dimensions]
        self.sourceTerms = zip(symbolics, values)
        return

    def updateSource(self, source, perturb=False):
        for index, value in enumerate(source):
            #if index == 1:
            #    phi = IOField.internalField('rhoUS', value, (3,))
            #    with IOField.handle(startTime):
            #        phi.write()
            if perturb:
                self.sourceTerms[index][1][:] += value
            else:
                self.sourceTerms[index][1][:] = value
        return

    def readFields(self, t):
        fields = []
        with IOField.handle(t):
            for name in self.names:
                fields.append(IOField.read(name))
        if self.firstRun:
            self.fields = fields
            for phi in self.fields:
                phi.completeField()
        else:
            self.updateFields(fields)
        self.firstRun = False
        return self.getFields(self.fields, IOField)
    
    def updateFields(self, fields):
        for phi, phiN in zip(self.fields, fields):
            phi.field = phiN.field
        for phi, phiB in zip(self.getBCFields(), [phi.boundary for phi in fields]):
            for patchID in self.mesh.sortedPatches:
                patch = phi.BC[patchID]
                for key, value in zip(patch.keys, patch.inputs):
                    if key == 'value':
                        nFaces = self.mesh.boundary[patchID]['nFaces']
                        value[1][:] = extractField(phiB[patchID][key], nFaces, phi.dimensions)
        return

    def getBCFields(self):
        return [phi.phi for phi in self.fields]

    def setBCFields(self, fields):
        for phi, phiN in zip(self.getBCFields(), fields):
            phi.field = phiN.field
        return

    def initFields(self, fields):
        newFields = self.init(*[phi.field for phi in fields])
        return self.getFields(newFields, IOField)

    def writeFields(self, fields, t, **kwargs):
        n = len(self.names)
        fields, rest = fields[:n], fields[n:]
        oldFields = fields
        fields = self.initFields(fields)
        for phi, phiN in zip(self.fields, fields):
            phi.field = phiN.field
        with IOField.handle(t):
            for phi in self.fields + rest:
                phi.write(**kwargs)
        return 

    def readStatusFile(self):
        data = None
        if parallel.rank == 0:
            with open(self.statusFile, 'rb') as status:
                readData = pkl.load(status)
            data = readData
        data = parallel.mpi.bcast(data, root=0)
        return data

    def writeStatusFile(self, data):
        if parallel.rank == 0:
            with open(self.statusFile, 'wb') as status:
                pkl.dump(data, status)
        return

    def removeStatusFile(self):
        if parallel.rank == 0:
            try:
                os.remove(self.statusFile)
            except OSError:
                pass

    def equation(self, *fields):
        pass

    def run(self, endTime=np.inf, writeInterval=config.LARGE, reportInterval=1, startTime=0.0, dt=1e-3, nSteps=config.LARGE, \
            startIndex=0, result=0., mode='simulation', source=lambda *args: [0.]*len(args[0]), perturbation=None, avgStart=0):

        logger.info('running solver for {0}'.format(nSteps))
        mesh = self.mesh
        mesh.reset = True
        #initialize
        fields = self.readFields(startTime)
        pprint()

        # time management
        t = startTime
        dts = dt
        timeIndex = startIndex
        if self.localTimeStep:
            dt = dts*np.ones_like(mesh.volumes)
        elif isinstance(dts, np.ndarray):
            dt = dts[timeIndex]

        # objective is local
        timeSeries = []
        timeSteps = []

        # writing and returning local solutions
        if mode == 'forward':
            if self.dynamicMesh:
                instMesh = Mesh()
                instMesh.boundary = copy.deepcopy(self.mesh.boundary)
                solutions = [[instMesh] + fields]
            else:
                solutions = [fields]


        def doPerturb(revert=False):
            parameters, perturb = perturbation
            values = perturb(fields, mesh, t)
            if not isinstance(values, list) or (len(parameters) == 1 and len(values) > 1):
                values = [values]
            for param, value in zip(parameters, values):
                if param == 'source':
                    if revert:
                        value = [-x for x in value]
                    pprint('Perturbing source')
                    self.updateSource(value, perturb=True)
                elif param == 'mesh':
                    pprint('Perturbing mesh')
                    for attr, delta in zip(Mesh.gradFields, value):
                        if revert:
                            delta = -delta
                        field = getattr(mesh, attr)
                        field += delta
                        assert field is getattr(mesh, attr)
                elif isinstance(param, tuple):
                    if revert:
                        value = -value
                    assert param[0] == 'BCs'
                    pprint('Perturbing', param)
                    _, phi, patchID, key = param
                    patch = getattr(self, phi).phi.BC[patchID]
                    index = patch.keys.index(key)
                    patch.inputs[index][1][:] += value
                else:
                    raise NotImplementedError

        # made static
        self.updateSource(source(fields, mesh, t))
        if perturbation:
            doPerturb()

        def iterate(t, timeIndex):
            return t < endTime and timeIndex < nSteps

        def updateTime(t0, dt):

            if self.localTimeStep:
                t = t0 + 1
            else:
                t = round(t0+dt, 12)
            return t


        pprint('Time step', timeIndex)
        for index in range(0, len(fields)):
            fields[index].info()
        pprint()

        while iterate(t, timeIndex):
            # add reporting interval
            mesh.reset = True
            report = ((timeIndex + 1) % reportInterval == 0) 
            write = ((timeIndex + 1) % writeInterval == 0) or not iterate(updateTime(t, dt), timeIndex+1)
            return_reusable = report or write or (mode == 'forward')
            replace_reusable = (timeIndex == startIndex)

            # source term update
            # perturbation

            pprint('Time step', timeIndex + 1)
            if report:
                pprint('Time marching for', ' '.join(self.names))
                start = time.time()

            inputs = [phi.field for phi in fields] + \
                     [np.array([[dt]], config.precision)] + \
                     mesh.getTensor() + mesh.getScalar() + \
                     [x[1] for x in self.sourceTerms] + \
                     self.getBoundaryTensor(1) + \
                     [x[1] for x in self.extraArgs]
            options = {'return_reusable': return_reusable,
                       'replace_reusable': replace_reusable
                      }

            start2 = time.time()
            outputs = self.map(*inputs, **options)
            pprint(time.time()-start2)
            newFields, dtc, objective = outputs[:3], outputs[3], outputs[4]
            objective = objective[0,0]
            dtc = dtc[0,0]
            fields = self.getFields(newFields, IOField, refFields=fields)

            if report:
                #print local.shape, local.dtype, (local).max(), (local).min(), np.isnan(local).any()
                #print remote.shape, remote.dtype, (remote).max(), (remote).min(), np.isnan(remote).any()
                #diff = local-remote
                #print diff.min(), diff.max()

                #local = IOField.internalField('local', local.reshape(-1,1), (1,))
                #with IOField.handle(t):
                #    local.write()
                #exit(1)

                for index in range(0, len(fields)):
                    fields[index].info()

                end = time.time()
                pprint('Time for iteration:', end-start)
                pprint('Time since beginning:', end-config.runtime)

                if self.localTimeStep:
                    pprint('Simulation Time:', t, 'Time step: min', parallel.min(dt), 'max', parallel.max(dt))
                else:
                    pprint('Simulation Time:', t, 'Time step:', dt)
            pprint()

            # time management
            timeSteps.append([t, dt])
            timeIndex += 1
            t = updateTime(t, dt)
            
            #print(t)
            if self.localTimeStep:
                dt = dtc
            elif isinstance(dts, np.ndarray):
                dt = dts[timeIndex]
            elif not self.fixedTimeStep:
                dt = min(parallel.min(2*self.CFL/dtc), dt*self.stepFactor, endTime-t)
                #dt = min(parallel.min(dtc), dt*self.stepFactor, endTime-t)
            if self.dynamicMesh:
                mesh.update(t, dt)

            # objective management
            if timeIndex > avgStart:
                result += objective
            timeSeries.append(objective)

            # write management
            if mode == 'forward':
                if self.dynamicMesh:
                    instMesh = Mesh()
                    instMesh.boundary = copy.deepcopy(self.mesh.boundary)
                    solutions.append([instMesh] + fields)
                else:
                    solutions.append(fields)
            elif write:
                # write mesh, fields, status
                if mode == 'orig' or mode == 'simulation':
                    #if len(dtc.shape) == 0:
                    #    dtc = dtc*np.ones((mesh.nInternalCells, 1))
                    #dtc = IOField.internalField('dtc', dtc, (1,))
                    # how do i do value BC patches?
                    #self.writeFields(fields + [dtc], t)
                    #self.writeFields(fields + [dtc, local], t)
                    self.writeFields(fields, t)

                # write timeSeries if in orig mode (problem.py)
                if parallel.rank == 0:
                    if mode == 'orig':
                        with open(self.timeStepFile, 'ab') as f:
                            np.savetxt(f, timeSteps)
                    if mode == 'orig' or mode == 'perturb':
                        with open(self.timeSeriesFile, 'ab') as f:
                            np.savetxt(f, timeSeries)
                timeSeries = []
                timeSteps = []
                self.writeStatusFile([timeIndex, t, dt, result])

        if perturbation:
            doPerturb(revert=True)

        if mode == 'forward':
            return solutions
        return result

