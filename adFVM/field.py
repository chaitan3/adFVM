import re
import os
import copy
import numpy as np
from numbers import Number
from contextlib import contextmanager

from . import config, parallel, BCs
from .parallel import pprint
from .mesh import extractField, writeField

from adpy.tensor import Variable

try:
    import h5py
except:
    pprint('h5py library not loaded')

logger = config.Logger(__name__)

class Field(object):
    @classmethod
    def setMesh(cls, mesh):
        cls.mesh = mesh

    @classmethod
    def setSolver(cls, solver):
        cls.setMesh(solver.mesh)
        cls.solver = solver

    def __init__(self, name, field, dimensions):
        self.name = name
        self.field = field
        self.dimensions = dimensions

    #def _getType(self):
    #    return np

    #def _getMesh(self):
    #    if isinstance(self.field, np.ndarray):
    #        return self.mesh
    #    else:
    #        return self.mesh

    # apply to ad
    #@classmethod
    #def max(self, a, b):
    #    return self('max({0},{1})'.format(a.name, b.name), ad.maximum(a.field, b.field), a.dimensions)
    #@classmethod
    #def min(self, a, b):
    #    return self('min({0},{1})'.format(a.name, b.name), ad.minimum(a.field, b.field), a.dimensions)

    #@classmethod
    #def switch(self, condition, a, b):
    #    return self('switch({0},{1})'.format(a.name, b.name), ad.where(condition, a.field, b.field), a.dimensions)

    #def stabilise(self, num):
    #    return self.__class__('stabilise({0})'.format(self.name), ad.where(self.field < 0., self.field - num, self.field + num), self.dimensions)

    #def sign(self):
    #    return self.__class__('abs({0})'.format(self.name), ad.sgn(self.field), self.dimensions)

    # apply to np
    def info(self):
        mesh = self.mesh
        pprint(self.name + ':', end='')
        # mesh values required outside theano
        field = self.field[:mesh.nLocalCells]
        nanCheck = np.logical_not(np.isfinite(field))
        if nanCheck.any():
            indices = np.where(nanCheck)[0]
            #indices += -mesh.nInternalCells + mesh.nInternalFaces
            print('rank:', parallel.rank, 
                  'internal cells', mesh.nInternalCells, 
                  'local cells:', mesh.nLocalCells, 
                  'indices', indices)
            print('FloatingPointException: NAN FOUND')
            with IOField.handle(10.0):
                IOField(self.name + '_nan', self.field, self.dimensions).write()
            parallel.mpi.Abort()
        fieldMin = parallel.min(field, allreduce=False)
        fieldMax = parallel.max(field, allreduce=False)
        pprint(' min:', fieldMin, 'max:', fieldMax)

    def copy(self):
        return self.__class__(self.name, self.field.copy(), self.dimensions)

    #def cross(self, phi):
    #    assert self.dimensions == (3,)
    #    assert phi.dimensions == (3,)
    #    product = np.cross(self.field, phi.field) 
    #    return self.__class__('cross({0},{1})'.format(self.name, phi.name), product, phi.dimensions)

    # apply to both np and ad

    # creates a view
    #def component(self, component): 
    #    assert self.dimensions == (3,)
    #    comp = ad.reshape(self.field[:,0], (-1,1))
    #    return self.__class__('{0}.{1}'.format(self.name, component), comp, (1,))

    def magSqr(self):
        assert self.dimensions == (3,)
        return self.__class__('magSqr({0})'.format(self.name), np.reshape(np.sum(self.field*self.field, axis=1), (-1,1)), (1,))

    #def sqrt(self):
    #    ad = self._getType()
    #    return self.__class__('abs({0})'.format(self.name), ad.sqrt(self.field), self.dimensions)

    #def mag(self):
    #    return self.magSqr().sqrt()

    #def sqr(self):
    #    return self.__class__('abs({0})'.format(self.name), self.field*self.field, self.dimensions)

    #def log(self):
    #    ad = self._getType()
    #    return self.__class__('log({0})'.format(self.name), ad.log(self.field), self.dimensions)

    #def exp(self):
    #    ad = self._getType()
    #    return self.__class__('exp({0})'.format(self.name), ad.exp(self.field), self.dimensions)


    #def abs(self):
    #    return self.__class__('abs({0})'.format(self.name), ad.abs(self.field), self.dimensions)

    #def dot(self, phi):
    #    assert phi.dimensions == (3,)
    #    ad = self._getType()
    #    # if tensor
    #    if len(self.dimensions) == 2:
    #        assert self.dimensions[1] == 3
    #        phi = self.__class__(phi.name, phi.field[:,np.newaxis,:], (1,3))
    #        dimensions = (self.dimensions[0],)
    #    else:
    #        assert self.dimensions == (3,)
    #        dimensions = (1,)
    #    product = ad.sum(self.field * phi.field, axis=-1)
    #    # if summed over vector
    #    if len(self.dimensions) == 1:
    #        #product = ad.reshape(product,(self.field.shape[0],1))
    #        product = ad.reshape(product,(-1,1))
    #    return self.__class__('dot({0},{1})'.format(self.name, phi.name), product, dimensions)

    #def dotN(self):
    #    return self.dot(self._getMesh().Normals)

    ## represents self * phi
    #def outer(self, phi):
    #    return self.__class__('outer({0},{1})'.format(self.name, phi.name), self.field[:,:,np.newaxis] * phi.field[:,np.newaxis,:], (3,3))
    #
    ## creates a view
    #def transpose(self):
    #    assert len(self.dimensions) == 2
    #    return self.__class__('{0}.T'.format(self.name), np.transpose(self.field, (0,2,1)), self.dimensions)

    #def trace(self):
    #    assert len(self.dimensions) == 2
    #    phi = self.field
    #    return self.__class__('tr({0})'.format(self.name), ad.reshape(phi[:,0,0] + phi[:,1,1] + phi[:,2,2], (-1,1)), (1,))

    #def norm(self):
    #    assert len(self.dimensions) == 2
    #    ad = self._getType()
    #    phi = self.field
    #    normPhi = ad.sqrt(ad.reshape(ad.sum(ad.sum(phi*phi, axis=2), axis=1), (-1,1)))
    #    return self.__class__('norm({0})'.format(self.name), normPhi, (1,))

    def __neg__(self):
        return self.__class__('-{0}'.format(self.name), -self.field, self.dimensions)

    def __mul__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}*{1}'.format(self.name, phi), self.field * phi, self.dimensions)
        else:
            product = self.field * phi.field
            dimensions = max(self.dimensions, phi.dimensions)
            return self.__class__('{0}*{1}'.format(self.name, phi.name), product, dimensions)


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

    def __truediv__(self, phi):
        return self.__div__(phi)

class CellField(Field):
    def __init__(self, name, field, dimensions, boundary={}):
        logger.debug('initializing CellField {0}'.format(name))
        super(self.__class__, self).__init__(name, field, dimensions)
        mesh = self.mesh

        if len(list(boundary.keys())) == 0:
            self.boundary = mesh.defaultBoundary
        else:
            self.boundary = boundary

        # CellField does not contain processor patch data, but the size is still full = nCells in original code
        self.BC = {}
        for patchID in mesh.sortedPatches:
            # skip processor patches
            patchType = self.boundary[patchID]['type']
            self.BC[patchID] = getattr(BCs, patchType)(self, patchID)

    def getTensor(self, index=0):
        mesh = self.mesh
        inputs = []
        for patchID in mesh.sortedPatches:
            inputs.extend([x[index] for x in self.BC[patchID].getInputs()])
        return inputs

    def updateGhostCells(self, phi):
        logger.info('updating ghost cells for {0}'.format(self.name))
        #patches = sorted(self.mesh.localPatches, key=lambda x: self.mesh.boundary[x]['startFace'])
        patches = self.mesh.sortedPatches
        #print phi
        for patchID in patches:
            #print patchID, self.BC[patchID]
            phi = self.BC[patchID].update(phi)
        return phi

class IOField(Field):
    _handle = None

    def __init__(self, name, field, dimensions, boundary={}, ghost=False):
        super(self.__class__, self).__init__(name, field, dimensions)
        logger.debug('initializing IOField {0}'.format(name))
        self.boundary = boundary
        if len(list(boundary.keys())) == 0:
            self.boundary = self.mesh.defaultBoundary
        self.phi = None
        if ghost:
            assert self.boundary == self.mesh.defaultBoundary
            self.defaultComplete()

    @classmethod
    def internalField(self, name, field, dimensions):
        phi = self(name, field, dimensions)  
        phi.partialComplete()
        return phi

    @classmethod
    def boundaryField(self, name, boundary, dimensions):
        mesh = self.mesh
        field = np.zeros((mesh.nCells,) + dimensions, config.precision)
        meshBoundary = copy.deepcopy(self.mesh.defaultBoundary)
        for patchID in boundary.keys():
            meshBoundary[patchID]['type'] = 'calculated'
            cellStartFace, cellEndFace, _ = mesh.getPatchCellRange(patchID)
            field[cellStartFace:cellEndFace] = boundary[patchID]

        return self(name, field, dimensions, meshBoundary)

    @classmethod
    def read(self, name, **kwargs):
        if config.hdf5:
            return self.readHDF5(name, **kwargs)
        else:
            return self.readFoam(name, **kwargs)

    
    @classmethod
    def readFoam(self, name, skipField=False):
        # mesh values required outside theano
        pprint('reading foam field {0}, time {1}'.format(name, self.time))
        timeDir = self._handle
        mesh = self.mesh
        try: 
            content = open(timeDir + name).read()
            foamFile = re.search(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), content).group(1)
            assert re.search('format[\s\t]+(.*?);', foamFile).group(1) == config.fileFormat
            vector = re.search('class[\s\t]+(.*?);', foamFile).group(1) == 'volVectorField'
            dimensions = 1 + vector*2
            bytesPerField = 8*(1 + 2*vector)
            startBoundary = content.find('boundaryField')
            if not skipField:
                data = re.search(re.compile('internalField[\s\r\n]+(.*)', re.DOTALL), content[:startBoundary]).group(1)
                internalField = extractField(data, mesh.nInternalCells, (dimensions,))
            else:
                internalField = np.zeros((0,) + (dimensions,), config.precision)
        except Exception as e:
            config.exceptInfo(e, (timeDir, name))

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
                if boundary[patchID]['type'] == 'cyclicAMI':
                    boundary[patchID]['type'] = 'cyclic'
            except Exception as e:
                config.exceptInfo(e, (timeDir + name, patchID))
        if vector:
            dimensions = (3,)
        else:
            dimensions = (1,)
                #import pdb;pdb.set_trace()
        #value = extractField(self.patch[key], nFaces, dimensions)
        return self(name, internalField, dimensions, boundary)

    @classmethod
    def readHDF5(self, name, skipField=False):
        pprint('reading hdf5 field {0}, time {1}'.format(name, self.time))

        assert self._handle is not None
        fieldsFile = self._handle
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

        mesh = self.mesh
        fieldData = fieldGroup['field']
        if skipField:
            delta = mesh.nInternalCells
        else:
            delta = 0
        with fieldData.collective:
            field = fieldData[parallelStart[0] + delta:parallelEnd[0]]
        field = np.array(field).astype(config.precision)
        dimensions = field.shape[1:]
        if not skipField:
            internalField = field[:mesh.nInternalCells]
        else:
            internalField = np.zeros((0,) + dimensions, config.precision)

        boundaryData = fieldGroup['boundary']
        with boundaryData.collective:
            boundaryList = boundaryData[parallelStart[1]:parallelEnd[1]]
        if config.py3:
            boundaryList = boundaryList.astype('U100')
        boundary = {}
        for patchID, key, value in boundaryList:
            if patchID not in boundary:
                boundary[patchID] = {}
            boundary[patchID][key] = value
        #print rank, name, parallelStart[1], parallelEnd[1], boundaryData
        for patchID in boundary:
            patch = boundary[patchID]
            if patch['type'] in BCs.valuePatches:
                cellStartFace, cellEndFace, _ = mesh.getPatchCellRange(patchID)
                patch['value'] = field[cellStartFace - delta:cellEndFace - delta]

        return self(name, internalField, dimensions, boundary)
 

    @classmethod
    def openHandle(self, time, case=None):
        timeDir = self.mesh.getTimeDir(time, case)
        if config.hdf5:
            self._handle = h5py.File(timeDir + '.hdf5', 'a', driver='mpio', comm=parallel.mpi)
        else:
            self._handle = timeDir + '/'
        self.time = time

    @classmethod
    def closeHandle(self):
        if config.hdf5:
            self._handle.close()
        self._handle = None
        self.time = None

    @classmethod
    @contextmanager
    def handle(self, time, case=None):
        self.openHandle(time, case=case)
        yield
        self.closeHandle()

    def getInternalField(self):
        return self.field[:self.mesh.nInternalCells]

    def getInternal(self):
        return self.__class__(self.name, self.getInternalField(), self.dimensions, self.boundary)

    def getPatch(self, patchID):
        cellStartFace, cellEndFace, _ = self.mesh.getPatchCellRange(patchID)
        return self.field[cellStartFace:cellEndFace]

    def defaultComplete(self):
        mesh = self.mesh
        field = np.zeros((mesh.nCells,) + self.dimensions)
        field[:mesh.nInternalCells] = self.field[:mesh.nInternalCells]
        for patchID in self.boundary:
            patch = self.boundary[patchID]
            startFace, endFace, cellStartFace, cellEndFace, nFaces = mesh.getPatchFaceCellRange(patchID)
            if patch['type'] == 'cyclic':
                neighbour = mesh.boundary[patchID]['neighbourPatch']
                startFace, endFace, _ = mesh.getPatchFaceRange(neighbour)
                internalCells = mesh.owner[startFace:endFace]
                field[cellStartFace:cellEndFace] = field[internalCells]
            elif patch['type'] == 'zeroGradient' or patch['type'] == 'empty' or patch['type'] == 'symmetryPlane': 
                internalCells = mesh.owner[startFace:endFace]
                field[cellStartFace:cellEndFace] = field[internalCells]
            else:
                assert patch['type'] in config.processorPatches
        self.field = parallel.getRemoteCells([field], self.mesh)[0]
        return

    def completeField(self):
        logger.debug('completing field {0}'.format(self.name))
        #return phiI
        self.phi = CellField(self.name, None, self.dimensions, self.boundary)
    
    def getTensor(self, *args, **kwargs):
        return self.phi.getTensor(*args, **kwargs)

    def partialComplete(self, value=0.):
        mesh = self.mesh
        boundary = self.boundary
        self.field = np.vstack((self.field, value*np.ones((mesh.nGhostCells,) + self.dimensions, config.precision)))
        for patchID in self.boundary:
            patch = boundary[patchID]
            startFace, endFace, cellStartFace, cellEndFace, nFaces = mesh.getPatchFaceCellRange(patchID)
            if patch['type'] in BCs.valuePatches:
                try:
                    value = extractField(patch['value'], nFaces, self.dimensions)
                    patch['value'] = value
                    self.field[cellStartFace:cellEndFace] = value
                except:
                    pass
            elif patch['type'] == 'cyclic':
                neighbour = mesh.boundary[patchID]['neighbourPatch']
                startFace, endFace, _ = mesh.getPatchFaceRange(neighbour)
                internalCells = mesh.owner[startFace:endFace]
                self.field[cellStartFace:cellEndFace] = self.field[internalCells]
            elif patch['type'] == 'zeroGradient':
                internalCells = mesh.owner[startFace:endFace]
                self.field[cellStartFace:cellEndFace] = self.field[internalCells]
   
    def write(self, name=None, skipProcessor=False):
        if name:
            self.name = name  
        if config.hdf5:
            return self.writeHDF5(skipProcessor)
        else:
            return self.writeFoam(skipProcessor)

    def writeFoam(self, skipProcessor=False):
        # mesh values required outside theano
        field = self.field
        if not skipProcessor:
            field = parallel.getRemoteCells([field], self.mesh)[0]

        # fetch processor information
        assert len(field.shape) == 2
        np.set_printoptions(precision=16)
        pprint('writing field {0}, time {1}'.format(self.name, self.time))

        mesh = self.mesh
        internalField = field[:mesh.nInternalCells]
        boundary = self.boundary
        for patchID in boundary:
            patch = boundary[patchID]
            # look into skipProcessor
            if patch['type'] in BCs.valuePatches:
                cellStartFace, cellEndFace, _ = mesh.getPatchCellRange(patchID)
                patch['value'] = field[cellStartFace:cellEndFace]
                
        self.writeFoamField(internalField, boundary)

    def writeFoamField(self, internalField, boundary, timeDir=None):
        name = self.name
        if timeDir is None:
            timeDir = self._handle
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
                if attr.startswith('_'):
                    continue
                if attr == 'value' and patch['type'] in BCs.valuePatches:
                    writeField(handle, patch[attr], dtype, 'value')
                else:
                    handle.write('\t\t' + attr + ' ' + patch[attr] + ';\n')
            handle.write('\t}\n')
        handle.write('}\n')
        handle.close()

    # nonuniform non-value inputs not supported (and fixedValue value)
    def writeHDF5(self, skipProcessor=False):
        # mesh values required outside theano
        pprint('writing hdf5 field {0}, time {1}'.format(self.name, self.time))

        boundary = []
        for patchID in self.boundary.keys():
            #print rank, self.name, patchID
            patch = self.boundary[patchID]
            for key, value in patch.items():
                if key.startswith('_'):
                    continue
                if not (key == 'value' and patch['type'] in BCs.valuePatches):
                    boundary.append([patchID, key, str(value)])
        boundary = np.array(boundary, dtype='S100')

        # fetch processor information
        field = self.field
        if not skipProcessor:
            field = parallel.getRemoteCells([field], self.mesh)[0]

        fieldsFile = self._handle
        fieldGroup = fieldsFile.require_group(self.name)

        parallelInfo = np.array([field.shape[0], boundary.shape[0]])
        parallelStart, parallelEnd, parallelSize = self.mesh.writeHDF5Parallel(fieldGroup, parallelInfo)

        fieldData = fieldGroup.require_dataset('field', (parallelSize[0],) + self.dimensions, np.float64)
        with fieldData.collective:
            fieldData[parallelStart[0]:parallelEnd[0]] = field.astype(np.float64)

        boundaryData = fieldGroup.require_dataset('boundary', (parallelSize[1], 3), 'S100') 
        with boundaryData.collective:
            boundaryData[parallelStart[1]:parallelEnd[1]] = boundary

        #fieldsFile.close()

    def decompose(self, time, data):
        assert parallel.nProcessors == 1
        mesh = self.mesh
        decomposed, addressing = data
        nprocs = len(decomposed)
        for i in range(0, nprocs):
            _, _, owner, _, boundary = decomposed[i]
            _, face, cell, _ = addressing[i]
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
                    indices = face[startFace:endFace]
                    if patch['type'] == 'processor':
                        cellIndices = mesh.neighbour[indices]
                        localIndices = owner[startFace:endFace]
                        revIndices = np.where(cell[localIndices] == cellIndices)[0]
                        cellIndices[revIndices] = mesh.owner[indices[revIndices]]
                        boundaryField[patchID]['value'] = self.field[cellIndices]
                    elif patch['type'] == 'processorCyclic':
                        referPatch = patch['referPatch']
                        neighbourPatch = mesh.boundary[referPatch]['neighbourPatch']
                        neighbourIndices = indices - mesh.boundary[referPatch]['startFace'] \
                                                   + mesh.boundary[neighbourPatch]['startFace'] 
                        neighbourIndices += - mesh.nInternalFaces + mesh.nInternalCells
                        boundaryField[patchID]['value'] = self.field[neighbourIndices]

            case = self.mesh.case + 'processor{}/'.format(i)
            self.writeFoamField(internalField, boundaryField, timeDir=self.mesh.getTimeDir(time, case=case) + '/')

        pprint('decomposing', self.name, 'to', nprocs, 'processors')
        pprint()
        return

    def interpolate(self, points):
        from scipy.interpolate import griddata
        mesh = self.mesh
        cellCentres = mesh.cellCentres[:mesh.nInternalCells]
        internalField = self.getInternalField()
        field = griddata(cellCentres[:,:2], internalField, points[:,:2])
        return field


#class ExchangerOp(T.Op):
#    __props__ = ()
#    def __init__(self):
#        if parallel.nProcessors == 1:
#            self.view_map = {0: [0]}
#
#    def make_node(self, x):
#        assert hasattr(self, '_props')
#        x = ad.as_tensor_variable(x)
#        return T.Apply(self, [x], [x.type()])
#
#    def perform(self, node, inputs, output_storage):
#        field = np.ascontiguousarray(inputs[0])
#        output_storage[0][0] = parallel.getRemoteCells(field, Field.mesh)
#
#    def grad(self, inputs, output_grads):
#        return [gradExchange(output_grads[0])]
#
#    def R_op(self, inputs, eval_points):
#        return [exchange(eval_points[0])]
#
#class FaceExchangerOp(T.Op):
#    __props__ = ()
#    def __init__(self):
#        if parallel.nProcessors == 1:
#            self.view_map = {0: [0]}
#
#    def make_node(self, x, y, z):
#        assert hasattr(self, '_props')
#        x = ad.as_tensor_variable(x)
#        y = ad.as_tensor_variable(y)
#        z = ad.as_tensor_variable(z)
#        return T.Apply(self, [x, y, z], [x.type()])
#
#    def perform(self, node, inputs, output_storage):
#        field1 = np.ascontiguousarray(inputs[0])
#        field2 = np.ascontiguousarray(inputs[1])
#        startFace = inputs[2]
#        output_storage[0][0] = parallel.getRemoteFaces(field1, field2, startFace, Field.mesh)
#
#    def grad(self, inputs, output_grads):
#        gradFace = gradFaceExchange(output_grads[0], inputs[2])
#        gradFace.append(T.gradient.disconnected_type())
#        return gradFace
#
#class gradFaceExchangerOp(T.Op):
#    __props__ = ()
#
#    def __init__(self):
#        if parallel.nProcessors == 1:
#            self.view_map = {0: [0]}
#
#    def make_node(self, x, y):
#        assert hasattr(self, '_props')
#        x = ad.as_tensor_variable(x)
#        y = ad.as_tensor_variable(y)
#        return T.Apply(self, [x, y], [x.type(), x.type()])
#
#    def perform(self, node, inputs, output_storage):
#        gradField = np.ascontiguousarray(inputs[0])
#        startFace = inputs[1]
#        gradField1, gradField2 = parallel.getAdjointRemoteFaces(gradField, startFace, Field.mesh)
#        output_storage[0][0] = gradField1
#        output_storage[1][0] = gradField2
#
#class gradExchangerOp(T.Op):
#    __props__ = ()
#
#    def __init__(self):
#        if parallel.nProcessors == 1:
#            self.view_map = {0: [0]}
#
#    def make_node(self, x):
#        assert hasattr(self, '_props')
#        x = ad.as_tensor_variable(x)
#        return T.Apply(self, [x], [x.type()])
#
#    def perform(self, node, inputs, output_storage):
#        field = np.ascontiguousarray(inputs[0])
#        output_storage[0][0] = parallel.getAdjointRemoteCells(field, Field.mesh)
#
#exchange = ExchangerOp()
#faceExchange = FaceExchangerOp()
#gradExchange = gradExchangerOp()
#gradFaceExchange = gradFaceExchangerOp()
