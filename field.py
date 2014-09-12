from __future__ import print_function
import numpy as np
from os import makedirs
from os.path import exists
from numbers import Number
import re

import BCs
from utils import ad, pprint
from utils import Logger, Exchanger
logger = Logger(__name__)
import utils

class Field(object):
    def __init__(self, name, mesh, field):
        self.name = name
        self.mesh = mesh
        self.field = field
        self.size = field.shape[0]
        self.dimensions = field.shape[1:]

    @classmethod
    def max(self, a, b):
        a_gt_b = ad.value(a.field) > ad.value(b.field)
        b_gt_a = 1 - a_gt_b
        return self('max({0},{1})'.format(a.name, b.name), a.mesh, a.field * ad.array(a_gt_b) + b.field * ad.array(b_gt_a))

    def info(self):
        pprint(self.name + ':', end='')
        pprint(' min:', utils.min(ad.value(self.field)), 'max:', utils.max(ad.value(self.field)))

    # creates a view
    def component(self, component): 
        assert self.dimensions == (3,)
        return self.__class__('{0}.{1}'.format(self.name, component), self.mesh, self.field[:, component].reshape((-1,1)))

    def magSqr(self):
        assert self.dimensions == (3,)
        return self.__class__('magSqr({0})'.format(self.name), self.mesh, ad.sum(self.field**2, axis=1).reshape((-1,1)))

    def mag(self):
        return self.magSqr()**0.5

    def abs(self):
        return self.__class__('abs({0})'.format(self.name), self.mesh, self.field * ad.array(2*((ad.value(self.field) > 0) - 0.5)))

    def dot(self, phi):
        assert self.dimensions[0] == 3
        # if tensor
        if len(self.dimensions) > 1:
            phi = self.__class__(phi.name, phi.mesh, phi.field[:,np.newaxis,:])
        product = ad.sum(self.field * phi.field, axis=-1)
        # if summed over vector
        if len(product.shape) == 1:
            product = product.reshape((-1,1))
        return self.__class__('dot({0},{1})'.format(self.name, phi.name), self.mesh, product)

    def dotN(self):
        return self.dot(self.mesh.Normals)

    def outer(self, phi):
        return self.__class__('outer({0},{1})'.format(self.name, phi.name), self.mesh, self.field[:,:,np.newaxis] * phi.field[:,np.newaxis,:])
    
    # creates a view
    def transpose(self):
        assert len(self.dimensions) == 2
        return self.__class__('{0}.T'.format(self.name), self.mesh, self.field.transpose((0,2,1)))

    def __neg__(self):
        return self.__class__('-{0}'.format(self.name), self.mesh, -self.field)

    def __mul__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}*{1}'.format(self.name, phi), self.mesh, self.field * phi)
        else:
            product = self.field * phi.field
            return self.__class__('{0}*{1}'.format(self.name, phi.name), self.mesh, self.field * phi.field)


    def __rmul__(self, phi):
        return self * phi

    def __pow__(self, power):
        return self.__class__('{0}**{1}'.format(self.name, power), self.mesh, self.field.__pow__(power))

    def __add__(self, phi):
        if isinstance(phi, Number):
            return self.__class__('{0}+{1}'.format(self.name, phi), self.mesh, self.field + phi)
        else:
            return self.__class__('{0}+{1}'.format(self.name, phi.name), self.mesh, self.field + phi.field)

    def __radd__(self, phi):
        return self.__add__(phi)

    def __sub__(self, phi):
        return self.__add__(-phi)

    def __div__(self, phi):
        return self.__class__('{0}/{1}'.format(self.name, phi.name), self.mesh, self.field / phi.field)

class CellField(Field):
    def __init__(self, name, mesh, field, boundary={}):
        logger.debug('initializing field {0}'.format(name))
        super(self.__class__, self).__init__(name, mesh, field)

        if len(list(boundary.keys())) == 0:
            self.boundary = mesh.defaultBoundary
        else:
            self.boundary = boundary

        if self.size == mesh.nInternalCells:
            self.field = ad.zeros((mesh.nCells,) + self.dimensions)

        self.BC = {}
        for patchID in self.boundary:
            # skip empty patches
            if mesh.boundary[patchID]['nFaces'] == 0:
                continue
            self.BC[patchID] = getattr(BCs, self.boundary[patchID]['type'])(self, patchID)

        if self.size == mesh.nInternalCells:
            self.setInternalField(field)

    @classmethod
    def zeros(self, name, mesh, dimensions):
        logger.info('initializing zeros field {0}'.format(name))
        return self(name, mesh, ad.zeros((mesh.nCells,) + dimensions))

    @classmethod
    def copy(self, phi):
        logger.info('copying field {0}'.format(phi.name))
        return self(phi.name, phi.mesh, ad.array(ad.value(phi.field).copy()), phi.boundary.copy())

    @classmethod
    def read(self, name, mesh, time):
        if time.is_integer():
            time = int(time)
        pprint('reading field {0}, time {1}'.format(name, time))
        timeDir = '{0}/{1}/'.format(mesh.case, time)

        content = open(timeDir + name).read()
        foamFile = re.search(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), content).group(1)
        vector = re.search('class[\s\t]+(.*?);', foamFile).group(1) == 'volVectorField'
        bytesPerField = 8*(1 + 2*vector)
        startBoundary = content.find('boundaryField')
        data = re.search(re.compile('internalField[\s\r\n]+(.*)', re.DOTALL), content[:startBoundary]).group(1)
        internalField = utils.extractField(data, mesh.nInternalCells, vector)
        content = content[startBoundary:]
        boundary = {}
        #field = re.search(re.compile('value[ ]+nonuniform[ ]+List<[a-z]>[\s\r\n]+\(', re.DOTALL), 'HEY;', content)
        #while field is not None:
        #    for patchID in mesh.boundary:
        #        nBytes = mesh.boundary[patchID]['nFaces']*bytesPerField
        #        exists = re.match('\)[\s\r\n]+;', content[field.end + 1 + nBytes:])
        #        if exists is not None:
        #            boundary[patchID] = 
        #    field = re.search(re.compile('value[ ]+nonuniform[ ]+List<[a-z]>[\s\r\n]+\(', re.DOTALL), 'HEY;', content)
        for patchID in mesh.boundary:
            patch = re.search(re.compile(patchID + '[\s\r\n]+{(.*?)}', re.DOTALL), content).group(1)
            boundary[patchID] = dict(re.findall(re.compile('[\n \t]*([a-zA-Z]+)[ ]+(.*?);', re.DOTALL), patch))
        return self(name, mesh, internalField, boundary)

    def write(self, time):
        if time.is_integer():
            time = int(time)
        assert len(self.dimensions) == 1
        np.set_printoptions(precision=16)
        pprint('writing field {0}, time {1}'.format(self.name, time))
        timeDir = '{0}/{1}/'.format(self.mesh.case, time)
        if not exists(timeDir):
            makedirs(timeDir)
        handle = open(timeDir + self.name, 'w')
        handle.write(utils.foamHeader)
        handle.write('FoamFile\n{\n')
        foamFile = utils.foamFile.copy()
        foamFile['object'] = self.name
        if self.dimensions[0] == 3:
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
        utils.writeField(handle, ad.value(self.getInternalField()), dtype, 'internalField')
        handle.write('boundaryField\n{\n')
        for patchID in self.boundary:
            handle.write('\t' + patchID + '\n\t{\n')
            patch = self.boundary[patchID]
            for attr in patch:
                handle.write('\t\t' + attr + ' ' + patch[attr] + ';\n')
            if patch['type'] in ['processor', 'calculated', 'processorCyclic']:
                utils.writeField(handle, self.BC[patchID].value, dtype, 'value')
            handle.write('\t}\n')
        handle.write('}\n')
        handle.close()

    def setInternalField(self, internalField):
        self.field[:self.mesh.nInternalCells] = internalField
        self.updateGhostCells()

    def getInternalField(self):
        return self.field[:self.mesh.nInternalCells]

    def updateGhostCells(self):
        logger.info('updating ghost cells for {0}'.format(self.name))
        exchanger = Exchanger()
        for patchID in self.BC:
            if self.boundary[patchID]['type'] in ['processor', 'processorCyclic']:
                self.BC[patchID].update(exchanger)
            else:
                self.BC[patchID].update()
        exchanger.wait()




