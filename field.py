import numpy as np
import h5py
from numbers import Number
import re
import os
import copy

import config, parallel
from config import ad, T
from parallel import pprint
import BCs
from mesh import extractField, writeField
logger = config.Logger(__name__)

class Field(object):
    @classmethod
    def setSolver(cls, solver):
        cls.solver = solver
        cls.setMesh(solver.mesh)
        # why is this needed?
        mesh = solver.mesh
        if not hasattr(mesh, 'Normals'):
            mesh.Normals = Field('nF', mesh.normals, (3,))

    @classmethod
    def setMesh(cls, mesh):
        cls.mesh = mesh

    def __init__(self, name, field, dimensions):
        self.name = name
        self.field = field
        self.dimensions = dimensions

    @classmethod
    def max(self, a, b):
        return self('max({0},{1})'.format(a.name, b.name), ad.maximum(a.field, b.field), a.dimensions)
    @classmethod
    def min(self, a, b):
        return self('min({0},{1})'.format(a.name, b.name), ad.minimum(a.field, b.field), a.dimensions)

    @classmethod
    def switch(self, condition, a, b):
        return self('switch({0},{1})'.format(a.name, b.name), ad.switch(condition, a.field, b.field), a.dimensions)


    def info(self):
        assert isinstance(self.field, np.ndarray)
        mesh = self.mesh.origMesh
        pprint(self.name + ':', end='')
        # mesh values required outside theano
        field = self.field[:mesh.nLocalCells]
        nanCheck = np.isnan(field)
        if nanCheck.any():
            print(parallel.rank, mesh.nInternalCells, mesh.nLocalCells, np.where(nanCheck)[0])
            raise FloatingPointError('nan found')
        fieldMin = parallel.min(field)
        fieldMax = parallel.max(field)
        pprint(' min:', fieldMin, 'max:', fieldMax)

    def getField(self, indices):
        if isinstance(indices, tuple):
            return self.__class__(self.name, self.field[indices[0]:indices[1]], self.dimensions)
        else:
            return self.__class__(self.name, self.field[indices], self.dimensions)

    def setField(self, indices, field):
        if isinstance(field, Field):
            field = field.field
        if isinstance(indices, tuple):
            self.field = ad.set_subtensor(self.field[indices[0]:indices[1]], field)
        else:
            self.field = ad.set_subtensor(self.field[indices], field)

    # creates a view
    def component(self, component): 
        assert self.dimensions == (3,)
        return self.__class__('{0}.{1}'.format(self.name, component), self.field[:,[0]], (1,))

    def magSqr(self):
        assert self.dimensions == (3,)
        if isinstance(self.field, np.ndarray):
            return self.__class__('magSqr({0})'.format(self.name), np.sum(self.field*self.field, axis=1, keepdims=True), (1,))
        else:
            return self.__class__('magSqr({0})'.format(self.name), ad.sum(self.field*self.field, axis=1, keepdims=True), (1,))

    def mag(self):
        return self.magSqr().sqrt()

    def sqrt(self):
        return self.__class__('abs({0})'.format(self.name), ad.sqrt(self.field), self.dimensions)

    def sqr(self):
        return self.__class__('abs({0})'.format(self.name), self.field*self.field, self.dimensions)

    def abs(self):
        return self.__class__('abs({0})'.format(self.name), ad.abs_(self.field), self.dimensions)

    def stabilise(self, num):
        return self.__class__('stabilise({0})'.format(self.name), ad.switch(ad.lt(self.field, 0.), self.field - num, self.field + num), self.dimensions)

    def sign(self):
        return self.__class__('abs({0})'.format(self.name), ad.sgn(self.field), self.dimensions)


    def dot(self, phi):
        assert self.dimensions[0] == 3
        # if tensor
        if len(self.dimensions) > 1:
            phi = self.__class__(phi.name, phi.field[:,np.newaxis,:], (1,3))
            dimensions = (3,)
        else:
            dimensions = (1,)
        product = ad.sum(self.field * phi.field, axis=-1)
        # if summed over vector
        if len(self.dimensions) == 1:
            product = product.reshape((self.field.shape[0],1))
        return self.__class__('dot({0},{1})'.format(self.name, phi.name), product, dimensions)

    def dotN(self):
        return self.dot(self.mesh.Normals)

    def cross(self, phi):
        assert self.dimensions == (3,)
        assert phi.dimensions == (3,)
        assert isinstance(self.field, np.ndarray)
        product = np.cross(self.field, phi.field) 
        return self.__class__('cross({0},{1})'.format(self.name, phi.name), product, phi.dimensions)

    # represents self * phi
    def outer(self, phi):
        return self.__class__('outer({0},{1})'.format(self.name, phi.name), self.field[:,:,np.newaxis] * phi.field[:,np.newaxis,:], (3,3))
    
    # creates a view
    def transpose(self):
        assert len(self.dimensions) == 2
        return self.__class__('{0}.T'.format(self.name), self.field.transpose((0,2,1)), self.dimensions)

    def trace(self):
        assert len(self.dimensions) == 2
        phi = self.field
        return self.__class__('tr({0})'.format(self.name), (phi[:,0,0] + phi[:,1,1] + phi[:,2,2]).reshape((phi.shape[0],1)), (1,))

    def norm(self):
        assert len(self.dimensions) == 2
        phi = self.field
        normPhi = ad.sqrt(ad.sum(ad.sum(phi*phi, axis=2), axis=1, keepdims=True))
        return self.__class__('norm({0})'.format(self.name), normPhi, (1,))

    def __neg__(self):
        return self.__class__('-{0}'.format(self.name), -self.field, self.dimensions)

    def __mul__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}*{1}'.format(self.name, phi), self.field * phi, self.dimensions)
        else:
            product = self.field * phi.field
            dimensions = max(self.dimensions, phi.dimensions)
            return self.__class__('{0}*{1}'.format(self.name, phi.name), self.field * phi.field, dimensions)


    def __rmul__(self, phi):
        return self * phi

    def __pow__(self, power):
        return self.__class__('{0}**{1}'.format(self.name, power), self.field.__pow__(power), self.dimensions)

    def __add__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}+{1}'.format(self.name, phi), self.field + phi, self.dimensions)
        else:
            return self.__class__('{0}+{1}'.format(self.name, phi.name), self.field + phi.field, self.dimensions)

    def __radd__(self, phi):
        return self.__add__(phi)

    def __sub__(self, phi):
        return self.__add__(-phi)

    def __div__(self, phi):
        return self.__class__('{0}/{1}'.format(self.name, phi.name), self.field / phi.field, self.dimensions)

class CellField(Field):
    def __init__(self, name, field, dimensions, boundary={}, ghost=False):
        logger.debug('initializing CellField {0}'.format(name))
        super(self.__class__, self).__init__(name, field, dimensions)
        mesh = self.mesh

        if len(list(boundary.keys())) == 0:
            self.boundary = mesh.defaultBoundary
        else:
            self.boundary = boundary

        # CellField does not contain processor patch data, but the size is still full = nCells in original code
        if ghost:
            # can be not filled
            self.resetField()

        # why initialize the boundary for ghost=False cases
        self.BC = {}
        for patchID in self.mesh.origPatches:
            # skip processor patches
            patchType = self.boundary[patchID]['type']
            self.BC[patchID] = getattr(BCs, patchType)(self, patchID)

        if ghost:
            self.setInternalField(field)

    @classmethod
    def copy(self, phi):
        logger.info('copying field {0}'.format(phi.name))
        return self(phi.name, phi.field.copy(), phi.dimensions, phi.boundary.copy())

    def resetField(self):
        mesh = self.mesh
        size = (mesh.nCells, ) + self.dimensions
        self.field = ad.bcalloc(config.precision(1.), size)
        #self.field.tag.test_value = np.zeros((mesh.origMesh.nCells,) + self.dimensions, config.precision)

    def setInternalField(self, internalField):
        self.setField((0, self.mesh.nInternalCells), internalField)
        self.updateGhostCells()

    def getInternalField(self):
        return self.field[:self.mesh.nInternalCells]

    def updateGhostCells(self):
        logger.info('updating ghost cells for {0}'.format(self.name))
        for patchID in self.BC:
            self.BC[patchID].update()
        self.field = exchange(self.field)

class IOField(Field):
    readWriteHandle = None

    def __init__(self, name, field, dimensions, boundary={}):
        super(self.__class__, self).__init__(name, field, dimensions)
        logger.debug('initializing IOField {0}'.format(name))
        self.boundary = boundary
        if len(list(boundary.keys())) == 0:
            self.boundary = self.mesh.defaultBoundary
        else:
            self.boundary = boundary
        self.func = None
        self.phi = None

    def complete(self):
        logger.debug('completing field {0}'.format(self.name))
        internalField = ad.matrix()
        #X.tag.test_value = self.field
        # CellField for later use
        self.phi = CellField(self.name, internalField, self.dimensions, self.boundary, ghost=True)
        return internalField

    def partialComplete(self):
        mesh = self.mesh.origMesh
        boundary = self.boundary
        self.field = np.vstack((self.field, np.zeros((mesh.nGhostCells,) + self.dimensions)))
        for patchID in self.boundary:
            patch = boundary[patchID]
            if patch['type'] in BCs.valuePatches:
                startFace = mesh.boundary[patchID]['startFace']
                nFaces = mesh.boundary[patchID]['nFaces']
                endFace = startFace + nFaces
                cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
                cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces
                try:
                    value = extractField(patch['value'], nFaces, self.dimensions)
                    self.field[cellStartFace:cellEndFace] = value
                except:
                    pass

    @classmethod
    def boundaryField(self, name, boundary, dimensions):
        mesh = self.mesh.origMesh
        field = np.zeros((mesh.nCells,) + dimensions, config.precision)
        meshBoundary = copy.deepcopy(self.mesh.defaultBoundary)
        for patchID in boundary.keys():
            meshBoundary[patchID]['type'] = 'calculated'
            startFace = mesh.boundary[patchID]['startFace']
            nFaces = mesh.boundary[patchID]['nFaces']
            endFace = startFace + nFaces
            cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
            cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces
            field[cellStartFace:cellEndFace] = boundary[patchID]

        return self(name, field, dimensions, meshBoundary)

    def getInternalField(self):
        return self.field[:self.mesh.origMesh.nInternalCells]

    @classmethod
    def openHandle(self, case, time):
        if config.hdf5:
            if time.is_integer():
                time = int(time)
            fieldsFile = case + str(time) + '.hdf5'
            IOField.readWriteHandle = h5py.File(fieldsFile, 'a', driver='mpio', comm=parallel.mpi)

    @classmethod
    def closeHandle(self):
        if config.hdf5:
            self.readWriteHandle.close()
            self.readWriteHandle = None

    @classmethod
    def read(self, name, mesh, time):
        if config.hdf5:
            return self.readHDF5(name, mesh, time)
        else:
            return self.readFoam(name, mesh, time)

    @classmethod
    def readFoam(self, name, mesh, time):
        # mesh values required outside theano
        pprint('reading foam field {0}, time {1}'.format(name, time))
        timeDir = mesh.getTimeDir(time)
        mesh = mesh.origMesh
        try: 
            content = open(timeDir + name).read()
            foamFile = re.search(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), content).group(1)
            assert re.search('format[\s\t]+(.*?);', foamFile).group(1) == config.fileFormat
            vector = re.search('class[\s\t]+(.*?);', foamFile).group(1) == 'volVectorField'
            dimensions = 1 + vector*2
            bytesPerField = 8*(1 + 2*vector)
            startBoundary = content.find('boundaryField')
            data = re.search(re.compile('internalField[\s\r\n]+(.*)', re.DOTALL), content[:startBoundary]).group(1)
            internalField = extractField(data, mesh.nInternalCells, (dimensions,))
        except Exception as e:
            config.exceptInfo(e, timeDir + name)

        content = content[startBoundary:]
        boundary = {}
        def getToken(x): 
            token = re.match('[\s\r\n\t]+([a-zA-Z0-9_\.\-\+<>\{\\/}]+)', x)
            return token.group(1), token.end()
        for patchID in mesh.boundary:
            try:
                patch = re.search(re.compile('[\s\r\n\t]+' + patchID + '[\s\r\n]+{', re.DOTALL), content)
                boundary[patchID] = {}
                start = patch.end()
                while 1:
                    key, end = getToken(content[start:])
                    start += end
                    if key == '}':
                        break
                    # skip non binary, non value, uniform or empty patches
                    elif key == 'value' and config.fileFormat == 'binary' and getToken(content[start:])[0] != 'uniform' and mesh.boundary[patchID]['nFaces'] != 0:
                        match = re.search(re.compile('[ ]+(nonuniform[ ]+List<[a-z]+>[\s\r\n\t0-9]*\()', re.DOTALL), content[start:])
                        nBytes = bytesPerField * mesh.boundary[patchID]['nFaces']
                        start += match.end()
                        prefix = match.group(1)
                        boundary[patchID][key] = prefix + content[start:start+nBytes]
                        start += nBytes
                        match = re.search('\)[\s\r\n\t]*;', content[start:])
                        boundary[patchID][key] += match.group(0)[:-1]
                        start += match.end()
                    else:
                        match = re.search(re.compile('[ ]+(.*?);', re.DOTALL), content[start:])
                        start += match.end() 
                        boundary[patchID][key] = match.group(1)
            except Exception as e:
                config.exceptInfo(e, (timeDir + name, patchID))
        if vector:
            dimensions = (3,)
        else:
            dimensions = (1,)

        return self(name, internalField, dimensions, boundary)

    @classmethod
    def readHDF5(self, name, mesh, time):

        pprint('reading hdf5 field {0}, time {1}'.format(name, time))

        assert IOField.readWriteHandle is not None
        fieldsFile = IOField.readWriteHandle
        fieldGroup = fieldsFile[name]
        parallelGroup = fieldGroup['parallel']

        rank = parallel.rank
        parallelStartData = parallelGroup['start']
        parallelEndData = parallelGroup['end']
        assert parallelStartData.shape[0] == parallel.nProcessors
        with parallelStartData.collective:
            parallelStart = parallelStartData[rank]
        with parallelEndData.collective:
            parallelEnd = parallelEndData[rank]

        mesh = mesh.origMesh
        fieldData = fieldGroup['field']
        with fieldData.collective:
            field = fieldData[parallelStart[0]:parallelEnd[0]]
        field = np.array(field).astype(config.precision)
        internalField = field[:mesh.nInternalCells]
        dimensions = field.shape[1:]

        boundaryData = fieldGroup['boundary']
        with boundaryData.collective:
            boundaryList = boundaryData[parallelStart[1]:parallelEnd[1]]
        boundary = {}
        for patchID, key, value in boundaryList:
            if patchID not in boundary:
                boundary[patchID] = {}
            boundary[patchID][key] = value
        #print rank, name, parallelStart[1], parallelEnd[1], boundaryData
        for patchID in boundary:
            patch = boundary[patchID]
            if patch['type'] in BCs.valuePatches:
                startFace = mesh.boundary[patchID]['startFace']
                nFaces = mesh.boundary[patchID]['nFaces']
                endFace = startFace + nFaces
                cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
                cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces
                patch['value'] = field[cellStartFace:cellEndFace]

        #fieldsFile.close()

        return self(name, internalField, dimensions, boundary)
    
    def write(self, time, skipProcessor=False):
        if config.hdf5:
            return self.writeHDF5(self.mesh.case, time, skipProcessor)
        else:
            return self.writeFoam(self.mesh.case, time, skipProcessor)

    def writeFoam(self, case, time, skipProcessor=False):
        # mesh values required outside theano
        if not skipProcessor:
            field = parallel.getRemoteCells(field, self.mesh)

        # fetch processor information
        assert len(field.shape) == 2
        np.set_printoptions(precision=16)
        pprint('writing field {0}, time {1}'.format(name, time))

        mesh = self.mesh.origMesh
        internalField = self.field[:mesh.nInternalCells]
        boundary = self.boundary
        for patchID in boundary:
            patch = boundary[patchID]
            # look into skipProcessor
            if patch['type'] in BCs.valuePatches:
                startFace = mesh.boundary[patchID]['startFace']
                nFaces = mesh.boundary[patchID]['nFaces']
                endFace = startFace + nFaces
                cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
                cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces
                patch['value'] = field[cellStartFace:cellEndFace]

        self.writeFoamField(case, time, internalField, boundary)

    def writeFoamField(self, case, time, internalField, boundary):
        name = self.name
        if time.is_integer():
            time = int(time)
        timeDir = '{0}/{1}/'.format(case, time)
        if not os.path.exists(timeDir):
            os.makedirs(timeDir)
        handle = open(timeDir + name, 'w')
        handle.write(config.foamHeader)
        handle.write('FoamFile\n{\n')
        foamFile = config.foamFile.copy()
        foamFile['object'] = name
        if internalField.shape[1] == 3:
            dtype = 'vector'
            foamFile['class'] = 'volVectorField'
        else:
            dtype = 'scalar'
            foamFile['class'] = 'volScalarField'
        for key in foamFile:
            handle.write('\t' + key + ' ' + foamFile[key] + ';\n')
        handle.write('}\n')
        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        handle.write('dimensions      [0 1 -1 0 0 0 0];\n')
        writeField(handle, internalField, dtype, 'internalField')
        handle.write('boundaryField\n{\n')
        for patchID in boundary:
            handle.write('\t' + patchID + '\n\t{\n')
            patch = boundary[patchID]
            for attr in patch:
                # look into skipProcessor
                if attr == 'value' and patch['type'] in BCs.valuePatches:
                    writeField(handle, patch[attr], dtype, 'value')
                else:
                    handle.write('\t\t' + attr + ' ' + patch[attr] + ';\n')
            handle.write('\t}\n')
        handle.write('}\n')
        handle.close()

    # nonuniform non-value inputs not supported (and fixedValue value)
    def writeHDF5(self, case, time, skipProcessor=False):
        # mesh values required outside theano
        pprint('writing hdf5 field {0}, time {1}'.format(self.name, time))
        nProcs = parallel.nProcessors
        rank = parallel.rank

        boundary = []
        for patchID in self.boundary.keys():
            #print rank, self.name, patchID
            patch = self.boundary[patchID]
            for key, value in patch.iteritems():
                if not (key == 'value' and patch['type'] in BCs.valuePatches):
                    boundary.append([patchID, key, str(value)])
        boundary = np.array(boundary, dtype='S100')

        # fetch processor information
        field = self.field
        if not skipProcessor:
            field = parallel.getRemoteCells(field, self.mesh)

        assert IOField.readWriteHandle is not None
            
        fieldsFile = IOField.readWriteHandle
        fieldGroup = fieldsFile.require_group(self.name)

        parallelInfo = np.array([field.shape[0], boundary.shape[0]])
        nInfo = len(parallelInfo)
        parallelStart = np.zeros_like(parallelInfo)
        parallelEnd = np.zeros_like(parallelInfo)
        parallel.mpi.Exscan(parallelInfo, parallelStart)
        parallel.mpi.Scan(parallelInfo, parallelEnd)
        parallelSize = parallelEnd.copy()
        parallel.mpi.Bcast(parallelSize, nProcs-1)

        parallelGroup = fieldGroup.require_group('parallel')
        parallelStartData = parallelGroup.require_dataset('start', (nProcs, nInfo), np.int64)
        parallelEndData = parallelGroup.require_dataset('end', (nProcs, nInfo), np.int64)
        with parallelStartData.collective:
            parallelStartData[rank] = parallelStart
        with parallelEndData.collective:
            parallelEndData[rank] = parallelEnd

        fieldData = fieldGroup.require_dataset('field', (parallelSize[0],) + self.dimensions, np.float64)
        with fieldData.collective:
            fieldData[parallelStart[0]:parallelEnd[0]] = field.astype(np.float64)

        boundaryData = fieldGroup.require_dataset('boundary', (parallelSize[1], 3), 'S100') 
        with boundaryData.collective:
            boundaryData[parallelStart[1]:parallelEnd[1]] = boundary

        #fieldsFile.close()

    def decompose(self, time, data):
        mesh = self.mesh.origMesh
        decomposed, addressing = data
        nprocs = len(decomposed)
        for i in range(0, nprocs):
            _, _, owner, _, boundary = decomposed[i]
            _, face, cell = addressing[i]
            internalField = self.field[cell]
            boundaryField = {}
            for patchID in boundary:
                patch = boundary[patchID]
                boundaryField[patchID] = {}
                if patchID in self.boundary:
                    for key in self.boundary[patchID]:
                        if (key == 'value') and (self.boundary[patchID]['type'] in BCs.valuePatches):
                            startFace = patch['startFace']
                            endFace = startFace + patch['nFaces']
                            indices = face[startFace:endFace]
                            cellIndices = indices - mesh.nInternalFaces + mesh.nInternalCells
                            boundaryField[patchID][key] = self.field[cellIndices]
                        else:
                            boundaryField[patchID][key] = self.boundary[patchID][key]
                else:
                    boundaryField[patchID]['type'] = patch['type']
                    startFace = patch['startFace']
                    endFace = startFace + patch['nFaces']
                    localIndices = owner[startFace:endFace]
                    indices = face[startFace:endFace]
                    cellIndices = mesh.neighbour[indices]
                    revIndices = np.where(cell[localIndices] == cellIndices)[0]
                    cellIndices[revIndices] = mesh.owner[indices[revIndices]]
                    boundaryField[patchID]['value'] = self.field[cellIndices]

            case = self.mesh.case + 'processor{}/'.format(i)
            self.writeFoamField(case, time, internalField, boundaryField)

        pprint('decomposing', self.name, 'to', nprocs, 'processors')
        return

class ExchangerOp(T.Op):
    __props__ = ()
    def __init__(self):
        if parallel.nProcessors == 1:
            self.view_map = {0: [0]}

    def make_node(self, x):
        assert hasattr(self, '_props')
        x = ad.as_tensor_variable(x)
        return T.Apply(self, [x], [x.type()])
    def perform(self, node, inputs, output_storage):
        field = np.ascontiguousarray(inputs[0])
        output_storage[0][0] = parallel.getRemoteCells(field, Field.mesh)

    def grad(self, inputs, output_grads):
        return [gradExchange(output_grads[0])]

    def R_op(self, inputs, eval_points):
        return [exchange(eval_points[0])]

class gradExchangerOp(T.Op):
    __props__ = ()

    def __init__(self):
        if parallel.nProcessors == 1:
            self.view_map = {0: [0]}

    def make_node(self, x):
        assert hasattr(self, '_props')
        x = ad.as_tensor_variable(x)
        return T.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        field = np.ascontiguousarray(inputs[0])
        output_storage[0][0] = parallel.getAdjointRemoteCells(field, Field.mesh)

exchange = ExchangerOp()
gradExchange = gradExchangerOp()
