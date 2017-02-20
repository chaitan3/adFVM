import numpy as np
import time
import cPickle as pkl
import os
import copy

from . import config, parallel, timestep
from .config import ad
from .parallel import pprint
from .memory import printMemUsage

from .field import Field, CellField, IOField
from .mesh import Mesh
from .mesh import extractField

logger = config.Logger(__name__)

class Solver(object):
    defaultConfig = {
                        'timeIntegrator': 'euler',
                        'objective': lambda *x: 0,
                        'adjoint': False,
                        'dynamicMesh': False,
                        'localTimeStep': False,
                        'fixedTimeStep': False,
                        'stepFactor': 1.0,
                        'postpro': [],
                        'sourceTerms': []
                    }

    def __init__(self, case, **userConfig):
        logger.info('initializing solver for {0}'.format(case))
        fullConfig = self.__class__.defaultConfig
        for key in userConfig:
            assert key in fullConfig
        fullConfig.update(userConfig)
        for key in fullConfig:
            setattr(self, key, fullConfig[key])

        if parallel.rank == 0:
            open(case + 'STARTED', 'w').close()
            print('touched', case + 'STARTED')
        self.mesh = Mesh.create(case)
        self.resultFile = self.mesh.case + 'objective.txt'
        self.statusFile = self.mesh.case + 'status.pkl'
        self.timeSeriesFile = self.mesh.case + 'timeSeries.txt'
        Field.setSolver(self)

        self.timeStepCoeff = getattr(timestep, self.timeIntegrator)()
        self.nStages = self.timeStepCoeff[0].shape[0]
        self.stage = 0
        self.init = None
        self.objective = None
        return

    def compile(self, adjoint=None):
        pprint('Compiling solver', self.__class__.defaultConfig['timeIntegrator'])
        mesh = self.mesh

        self.compileInit()

        if self.localTimeStep:
            self.dt = ad.placeholder(config.dtype)
        else:
            self.dt = ad.placeholder(config.dtype)
        self.t0 = ad.placeholder(config.dtype)

        # default values
        self.t = self.t0*1.
        self.dtc = self.dt
        self.local, self.remote = self.mesh.nCells, self.mesh.nFaces
        self.stage = 1

        fields = self.getSymbolicFields(False)
        initFields = self.boundary(*timestep.createFields(fields, self))
        newFields = timestep.timeStepper(self.equation, self.boundary, initFields, self)
        if self.objective:
            objective = self.objective(initFields, mesh)
        else:
            objective = ad.constant(0.)
        self.map = self.function(fields + [self.dt, self.t0], \
                                 newFields + [objective, self.dtc, self.local, self.remote], 'forward')

        if adjoint:
            objGrad = [phi/mesh.volumes for phi in ad.gradients(objective, fields)]
            adjointFields = self.getSymbolicFields(False)
            gradientInputs = fields + adjoint.getGradFields()
            scalarFields = sum([ad.sum(newFields[index]*adjointFields[index]*mesh.volumes) \
                                for index in range(0, len(fields))])
            gradients = list(ad.gradients(scalarFields, gradientInputs)) + objGrad
            for index in range(0, len(fields)):
                gradients[index] /= mesh.volumes
            self.gradient = self.function(fields + adjointFields + [self.dt, self.t0], \
                            gradients, 'adjoint')
            #self.tangent = self.function([stackedFields, stackedAdjointFields, self.dt], \
            #                ad.Rop(newStackedFields, stackedFields, stackedAdjointFields), 'tangent')
        #if config.compile:
        #    exit()
        pprint()

    def initOrder(self, fields):
        return fields

    def compileInit(self, functionName='init'):
        internalFields = []
        completeFields = []
        for phi in self.fields:
            phiI = phi.completeField()
            internalFields.append(phiI)
            #completeFields.append(phiN)
        for phi, phiI in zip(self.initOrder(self.fields), self.initOrder(internalFields)):
            phi.phi.setInternalField(phiI)
        for phi in self.fields:
            phiN = phi.phi.field
            completeFields.append(phiN)
        self.init = self.function(internalFields, completeFields, functionName, source=False, postpro=False)
        return

    def function(self, inputs, outputs, name, **kwargs):
        return SolverFunction(inputs, outputs, self, name, **kwargs)

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

    #def stackFields(self, fields, mod): 
    #    return mod.concatenate([phi.field for phi in fields], axis=1)

    #def unstackFields(self, stackedFields, mod, names=None, **kwargs):
    #    if names is None:
    #        names = self.names
    #    fields = []
    #    nDimensions = np.concatenate(([0], np.cumsum(np.array(self.dimensions))))
    #    nDimensions = zip(nDimensions[:-1], nDimensions[1:])
    #    for name, dim, dimRange in zip(names, self.dimensions, nDimensions):
    #        phi = stackedFields[:, range(*dimRange)]
    #        fields.append(mod(name, phi, dim, **kwargs))
    #    return fields

    def getSymbolicFields(self, returnField=True):
        names = self.names
        fields = []
        for index, dim in enumerate(self.dimensions):
            if dim == (1,):
                field = ad.placeholder(config.dtype)
            else:
                field = ad.placeholder(config.dtype)
            if returnField:
                field = CellField(names[index], field, dim)
            fields.append(field)
        return fields

    def initSource(self):
        self.sourceFields = self.getSymbolicFields()
        symbolics = [phi.field for phi in self.sourceFields]
        values = [np.zeros((self.mesh.origMesh.nInternalCells, nDims[0]), config.precision) for nDims in self.dimensions]
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
        if self.init is None:
            self.fields = fields
        else:
            self.updateFields(fields)
        return self.getFields(self.fields, IOField)
    
    def updateFields(self, fields):
        for phi, phiN in zip(self.fields, fields):
            phi.field = phiN.field
        for phi, phiB in zip(self.getBCFields(), [phi.boundary for phi in fields]):
            for patchID in self.mesh.sortedPatches:
                patch = phi.BC[patchID]
                for key, value in zip(patch.keys, patch.inputs):
                    if key == 'value':
                        nFaces = self.mesh.origMesh.boundary[patchID]['nFaces']
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
        fields = self.initFields(fields)
        for phi, phiN in zip(self.fields, fields):
            phi.field = phiN.field
        with IOField.handle(t):
            for phi in self.fields + rest:
                phi.write(**kwargs)
        return 

    def readStatusFile(self):
        data = None
        scatterData = None
        if parallel.rank == 0:
            with open(self.statusFile, 'rb') as status:
                readData = pkl.load(status)
            data = readData[:-1]
            scatterData = readData[-1]
        data = parallel.mpi.bcast(data, root=0)
        data.append(parallel.mpi.scatter(scatterData, root=0))
        return data

    def writeStatusFile(self, data):
        data[-1] = parallel.mpi.gather(data[-1], root=0)
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

    def boundary(self, *newFields):
        boundaryFields = self.getBCFields()
        for phiB, phiN in zip(boundaryFields, newFields):
            phiB.resetField()
            phiB.setInternalField(phiN.field)
        return boundaryFields

    def run(self, endTime=np.inf, writeInterval=config.LARGE, reportInterval=1, startTime=0.0, dt=1e-3, nSteps=config.LARGE, \
            startIndex=0, result=0., mode='simulation', source=lambda *args: [0.]*len(args[0]), perturbation=None):

        logger.info('running solver for {0}'.format(nSteps))
        mesh = self.mesh
        mesh.reset = True
        # TODO: re-read optimization
        #initialize
        fields = self.readFields(startTime)
        pprint()

        # time management
        t = startTime
        dts = dt
        timeIndex = startIndex
        if self.localTimeStep:
            dt = dts*np.ones_like(mesh.origMesh.volumes)
        elif isinstance(dts, np.ndarray):
            dt = dts[timeIndex]

        # objective is local
        timeSeries = []
        timeSteps = []

        # writing and returning local solutions
        if mode == 'forward':
            if self.dynamicMesh:
                instMesh = Mesh()
                instMesh.boundary = copy.deepcopy(self.mesh.origMesh.boundary)
                solutions = [[instMesh] + fields]
            else:
                solutions = [fields]


        def doPerturb(revert=False):
            parameters, perturb = perturbation
            values = perturb(fields, mesh.origMesh, t)
            if not isinstance(values, list) or (len(parameters) == 1 and len(values) > 1):
                values = [values]
            for param, value in zip(parameters, values):
                if param == 'source':
                    if revert:
                        value = -value
                    pprint('Perturbing source')
                    self.updateSource(value, perturb=True)
                elif param == 'mesh':
                    pprint('Perturbing mesh')
                    for attr, delta in zip(Mesh.gradFields, value):
                        if revert:
                            delta = -delta
                        field = getattr(mesh.origMesh, attr)
                        field += delta
                        assert field is getattr(mesh.origMesh, attr)
                elif isinstance(param, tuple):
                    if revert:
                        value = -value
                    assert param[0] == 'BCs'
                    pprint('Perturbing', param)
                    _, phi, patchID, key = param
                    patch = getattr(self, phi).phi.BC[patchID]
                    index = patch.keys.index(key)
                    patch.inputs[index][1][:] += value
                elif isinstance(param, ad.TensorType):
                    raise NotImplementedError
                pprint('Time marching for', ' '.join(self.names))

        def iterate(t, timeIndex):
            return t < endTime and timeIndex < nSteps
        while iterate(t, timeIndex):
            # add reporting interval
            mesh.reset = True
            report = (timeIndex % reportInterval) == 0

            # source term update
            self.updateSource(source(fields, mesh.origMesh, t))
            # perturbation
            if perturbation:
                doPerturb()

            if report:
                printMemUsage()
                start = time.time()
                for index in range(0, len(fields)):
                    fields[index].info()
                pprint('Time step', timeIndex)
                #try:
                #    fields[index].info()
                #except:
                #    with IOField.handle(t):
                #        fields[index].write()
                #    exit(1)

            inputs = fields + [dt, t]
            outputs = self.map(*inputs)
            newFields, objective, dtc, local, remote = outputs[:-4], outputs[-4], outputs[-3], outputs[-2], outputs[-1]
            fields = self.getFields(newFields, IOField, refFields=fields)

            if report:
                #print local.shape, local.dtype, (local).max(), (local).min(), np.isnan(local).any()
                #print remote.shape, remote.dtype, (remote).max(), (remote).min(), np.isnan(remote).any()
                pprint('Percent shock capturing: {0:.2f}%'.format(float(parallel.max(local))*100))
                #diff = local-remote
                #print diff.min(), diff.max()

                #local = IOField.internalField('local', local.reshape(-1,1), (1,))
                #with IOField.handle(t):
                #    local.write()
                #exit(1)


                #parallel.mpi.Barrier()
                end = time.time()
                pprint('Time for iteration:', end-start)
                pprint('Time since beginning:', end-config.runtime)
                pprint('Running average objective: ', parallel.sum(result)/(timeIndex + 1))

                if self.localTimeStep:
                    pprint('Simulation Time:', t, 'Time step: min', parallel.min(dt), 'max', parallel.max(dt))
                else:
                    pprint('Simulation Time:', t, 'Time step:', dt)
                pprint()

            # time management
            timeSteps.append([t, dt])
            timeIndex += 1
            if self.localTimeStep:
                t += 1
            else:
                t = round(t+dt, 9)
            if self.localTimeStep:
                dt = dtc
            elif isinstance(dts, np.ndarray):
                dt = dts[timeIndex]
            elif not self.fixedTimeStep:
                dt = min(parallel.min(dtc), dt*self.stepFactor, endTime-t)

            if self.dynamicMesh:
                mesh.update(t, dt)

            # objective management
            result += objective
            timeSeries.append(parallel.sum(objective))

            # write management
            if mode == 'forward':
                if self.dynamicMesh:
                    instMesh = Mesh()
                    instMesh.boundary = copy.deepcopy(self.mesh.origMesh.boundary)
                    solutions.append([instMesh] + fields)
                else:
                    solutions.append(fields)
            elif (timeIndex % writeInterval == 0) or not iterate(t, timeIndex):
                # write mesh, fields, status
                if mode == 'orig' or mode == 'simulation':
                    if len(dtc.shape) == 0:
                        dtc = dtc*np.ones((mesh.origMesh.nInternalCells, 1))
                    dtc = IOField.internalField('dtc', dtc, (1,))
                    # how do i do value BC patches?
                    self.writeFields(fields + [dtc], t)
                    #self.writeFields(fields + [dtc, local], t)

                # write timeSeries if in orig mode (problem.py)
                if parallel.rank == 0:
                    lastIndex = timeIndex - (startIndex + writeInterval)
                    if mode == 'orig':
                        with open(self.timeStepFile, 'a') as f:
                            np.savetxt(f, timeSteps[lastIndex:])
                    if mode == 'orig' or mode == 'perturb':
                        with open(self.timeSeriesFile, 'a') as f:
                            np.savetxt(f, timeSeries[lastIndex:])
                self.writeStatusFile([timeIndex, t, dt, result])

        if mode == 'forward':
            return solutions
        return result


class SolverFunction(object):
    counter = 0
    def __init__(self, inputs, outputs, solver, name, BCs=True, source=True, postpro=True):
        logger.info('compiling function')
        self.symbolic = []
        self.values = []
        self.name = name
        mesh = solver.mesh
        # values require inplace substitution
        self.populate_mesh(self.symbolic, mesh, mesh)
        self.populate_mesh(self.values, mesh.origMesh, mesh)
        if BCs:
            self.populate_BCs(self.symbolic, solver, 0)
            self.populate_BCs(self.values, solver, 1)
        # source terms
        if source and len(solver.sourceTerms) > 0:
            symbolic, values = zip(*solver.sourceTerms)
            self.symbolic.extend(symbolic)
            self.values.extend(values)
        # postpro variables
        if postpro and len(solver.postpro) > 0:
            symbolic, values = zip(*solver.postpro)
            self.symbolic.extend(symbolic)
            self.values.extend(values)

        self.generate(inputs, outputs, solver.mesh.case)

    def populate_mesh(self, inputs, mesh, solverMesh):
        attrs = Mesh.fields + Mesh.constants
        for attr in attrs:
            val = getattr(mesh, attr)
            if attr == 'sumOp' or attr == 'gradOp':
                inputs.extend([val.indices, val.values, val.dense_shape])
            else:
                inputs.append(val)
        for patchID in solverMesh.sortedPatches:
            for attr in solverMesh.getBoundaryTensor(patchID):
                inputs.append(mesh.boundary[patchID][attr[0]])

    def populate_BCs(self, inputs, solver, index):
        fields = solver.getBCFields()
        for phi in fields:
            if hasattr(phi, 'BC'):
                for patchID in solver.mesh.sortedPatches:
                    inputs.extend([value[index] for value in phi.BC[patchID].inputs])

    def generate(self, inputs, outputs, caseDir):
        SolverFunction.counter += 1
        pklFile = caseDir + '{0}_func_{1}.pkl'.format(config.device, self.name)
        inputs = inputs + self.symbolic

        fn = None
        pklData = None
        start = time.time()
        if 0:#os.path.exists(pklFile) and config.unpickleFunction:
            pprint('Loading pickled file', pklFile)
            pklData = open(pklFile).read()
        else:
            #fn = ad.function(inputs, outputs, on_unused_input='ignore', mode=config.compile_mode)#, allow_input_downcast=True)
            fn = (inputs, outputs)
            #T.printing.pydotprint(fn, outfile=name + '_graph.png')
            #if config.pickleFunction or (parallel.nProcessors > 1):
            #pklData = pkl.dumps(fn)
            #if config.pickleFunction:
            #    pprint('Saving pickle file', pklFile)
            #    open(pklFile, 'w').write(pklData)
            #    pprint('Module size: {0:.2f}'.format(float(len(pklData))/(1024*1024)))
        end = time.time()
        pprint('Compilation time for {}: {:.2f}'.format(self.name, end-start))

        #if not config.compile:
        #    start = time.time()
        #    pklData = parallel.mpi.bcast(pklData, root=0)
        #    #if parallel.mpi.bcast(fn is not None, root=0) and parallel.nProcessors > 1:
        #    #    T.gof.cc.get_module_cache().refresh(cleanup=False)
        #    end = time.time()
        #    pprint('Transfer time: {0:.2f}'.format(end-start))

        #    start = time.time()
        #    unloadingStages = config.user.unloadingStages
        #    coresPerNode = config.user.coresPerNode
        #    coresPerStage = coresPerNode/unloadingStages
        #    nodeStage = (parallel.rank % coresPerNode)/coresPerStage
        #    for stage in range(unloadingStages):
        #        printMemUsage()
        #        if (nodeStage == stage) and (fn is None):
        #            fn = pkl.loads(pklData)
        #        parallel.mpi.Barrier()
        #    end = time.time()
        #    pprint('Loading time: {0:.2f}'.format(end-start))
        #    printMemUsage()

        self.fn = fn
        self.sess = ad.Session()
        #init = ad.global_variables_initializer()
        #self.sess.run(init)
        #config = ad.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8, \

        #                                allow_soft_placement=True, device_count = {'CPU': 8})
        #with ad.Session(config=config) as sess: 

    def __call__(self, *inputs):
        logger.info('running function')
        #pprint('running function', self.name)
        inputs = list(inputs)
        for index, inp in enumerate(inputs):
            if isinstance(inp, Field):
                inputs[index] = inp.field
                
        inputs = inputs + self.values
        #print 'get', id(self.values[29].data)
        
        #run_options = ad.RunOptions(trace_level=ad.RunOptions.FULL_TRACE)
        #run_metadata = ad.RunMetadata()

        inp, out = self.fn
        feed_dict = {inp[i]:inputs[i] for i in range(0, len(inp))}
        outputs = self.sess.run(out, feed_dict=feed_dict)#, options=run_options, run_metadata=run_metadata)
        #from tensorflow.python.client import timeline
        #tl = timeline.Timeline(run_metadata.step_stats)
        #ctf = tl.generate_chrome_trace_format()
        #with open('timeline.json', 'w') as f:
        #    f.write(ctf)
        if isinstance(outputs, tuple):
            outputs = list(outputs)
        return outputs

