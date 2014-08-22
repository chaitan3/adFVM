from __future__ import print_function
import numpy as np
import numpad as ad
from os import makedirs
from os.path import exists
import re
import numbers

import BCs
import utils
logger = utils.logger(__name__)

class Field(object):
    def __init__(self, name, mesh, field):
        self.name = name
        self.mesh = mesh
        self.field = field
        self.size = field.shape[0]
        self.dimensions = field.shape[1:]

        if self.dimensions not in [(1,), (3,), (3,3), (1,3), (3,1), (1,1)]:
            raise Exception('field dimensions {0} not supported'.format(self.dimensions))

    @classmethod
    def max(self, a, b):
        a_gt_b = ad.value(a.field) > ad.value(b.field)
        b_gt_a = 1 - a_gt_b
        return self('max({0},{1})'.format(a.name, b.name), a.mesh, a.field * ad.array(a_gt_b) + b.field * ad.array(b_gt_a))

    def info(self):
        print(self.name + ':', self.field.shape, end='')
        #print(max(np.isnan(self.field._value.tolist())))
        #print(np.where(self.field._value < 0))
        print(' min:', ad.value(self.field).min(), 'max:', ad.value(self.field).max())

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
    
    def transpose(self):
        assert len(self.dimensions) == 2
        return self.__class__('{0}.T'.format(self.name), self.mesh, self.field.transpose((0,2,1)))

    def __neg__(self):
        return self.__class__('-{0}'.format(self.name), self.mesh, -self.field)

    def __mul__(self, phi):
        if isinstance(phi, numbers.Number):
            return self.__class__('{0}*{1}'.format(self.name, phi), self.mesh, self.field * phi)
        else:
            product = self.field * phi.field
            return self.__class__('{0}*{1}'.format(self.name, phi.name), self.mesh, self.field * phi.field)


    def __rmul__(self, phi):
        return self * phi

    def __pow__(self, power):
        return self.__class__('{0}**{1}'.format(self.name, power), self.mesh, self.field.__pow__(power))

    def __add__(self, phi):
        if isinstance(phi, numbers.Number):
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
        return self(phi.name, phi.mesh, phi.field.copy(), phi.boundary.copy())

    @classmethod
    def read(self, name, mesh, time):
        if time == 0.0:
            time = 0
        print('reading field {0}, time {1}'.format(name, time))
        timeDir = '{0}/{1}/'.format(mesh.case, time)

        content = utils.removeCruft(open(timeDir + name, 'r').read(), keepHeader=True)
        foamFile = re.search(re.compile('FoamFile\n{(.*?)}\n', re.DOTALL), content).group(1)
        vector = re.search('class[\s\t]+(.*?);', foamFile).group(1) == 'volVectorField'
        data = re.search(re.compile('internalField[\s\r\n]+(.*?);', re.DOTALL), content).group(1)
        internalField = utils.extractField(data, mesh.nInternalCells, vector)
        content = content[content.find('boundaryField'):]
        boundary = {}
        for patchID in mesh.boundary:
            patch = re.search(re.compile(patchID + '[\s\r\n]+{(.*?)}', re.DOTALL), content).group(1)
            boundary[patchID] = dict(re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(.*?);', patch))
        return self(name, mesh, internalField, boundary)

    def write(self, time):
        if time == 0.0:
            time = 0
        assert len(self.dimensions) == 1
        np.set_printoptions(precision=16)
        print('writing field {0}, time {1}'.format(self.name, time))
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

        handle.write('internalField   nonuniform List<'+ dtype +'>\n')
        handle.write('{0}\n(\n'.format(self.mesh.nInternalCells))
        for value in ad.value(self.getInternalField()):
            if dtype == 'scalar':
                handle.write(str(value[0]) + '\n')
            else:
                handle.write('(' + ' '.join(np.char.mod('%f', value)) + ')\n')
        handle.write(')\n;\n')
        handle.write('boundaryField\n{\n')
        for patchID in self.boundary:
            handle.write('\t' + patchID + '\n\t{\n')
            patch = self.boundary[patchID]
            for attr in patch:
                handle.write('\t\t' + attr + ' ' + patch[attr] + ';\n')
            handle.write('\t}\n')
        handle.write('}\n')
        handle.close()

    def setInternalField(self, internalField):
        mesh = self.mesh
        self.field[:mesh.nInternalCells] = internalField
        self.updateGhostCells()

    def getInternalField(self):
        mesh = self.mesh
        return self.field[:mesh.nInternalCells]

    def updateGhostCells(self):
        logger.info('updating ghost cells for {0}'.format(self.name))
        mesh = self.mesh
        for patchID in mesh.boundary:
            self.BC[patchID].update()



