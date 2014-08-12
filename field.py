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

class FaceField:
    def __init__(self, name, mesh, field):
        self.name = name
        self.mesh = mesh
        self.field = field

    def mag(self):
        return FaceField(self.name, self.mesh, ad.sum(self.field, axis=-1).reshape((-1,1)))

    def __neg__(self):
        return FaceField(self.name, self.mesh, -self.field)

    def __mul__(self, field):
        if isinstance(field, numbers.Number):
            return FaceField(self.name, self.mesh, self.field * field)
        else:
            return FaceField(self.name, self.mesh, self.field * field.field)

    def __rmul__(self, field):
        return self * field

    def __add__(self, field):
        return FaceField(self.name, self.mesh, self.field + field.field)

    def __sub__(self, field):
        return self.__add__(-field)

    def __div__(self, field):
        return FaceField(self.name, self.mesh, self.field / field.field)

class Field(FaceField):
    def __init__(self, name, mesh, internalField, boundary={}):
        logger.info('initializing field {0}'.format(name))
        self.name = name
        self.mesh = mesh
        self.field = ad.zeros((mesh.nCells, internalField.shape[1]))
        self.boundary = boundary
        self.setInternalField(internalField)

    @classmethod
    def zeros(self, name, mesh, dimensions):
        logger.info('initializing zeros field {0}'.format(name))
        boundary = {}
        for patch in mesh.boundary:
            boundary[patch] = {}
            if mesh.boundary[patch]['type'] == 'cyclic':
                boundary[patch]['type'] = 'cyclic'
            else:
                boundary[patch]['type'] = 'zeroGradient'
        return self(name, mesh, ad.zeros((mesh.nInternalCells, dimensions)), boundary)

    @classmethod
    def copy(self, field):
        logger.info('copying field {0}'.format(field.name))
        return self(field.name, field.mesh, field.getInternalField(), field.boundary.copy())

    @classmethod
    def read(self, name, mesh, time):
        print('reading field {0}, time {1}\n'.format(name, time))
        timeDir = '{0}/{1}/'.format(mesh.case, time)
        content = utils.removeCruft(open(timeDir + name, 'r').read())

        data = re.search(re.compile('internalField[\s\r\na-zA-Z<>]+(.*?);', re.DOTALL), content).group(1)
        start = data.find('(')
        if start >= 0:
            internalField = ad.adarray(re.findall('[0-9\.Ee\-]+', data[start:])).reshape((-1,1))
        else:
            internalField = ad.ones((mesh.nInternalCells, 1))

        content = content[content.find('boundaryField'):]
        boundary = {}
        for patchID in mesh.boundary:
            patch = re.search(re.compile(patchID + '[\s\r\n]+{(.*?)}', re.DOTALL), content).group(1)
            boundary[patchID] = dict(re.findall('\n[ \t]+([a-zA-Z]+)[ ]+(.*?);', patch))
        return self(name, mesh, internalField, boundary)

    def write(self, time):
        print('writing field {0}, time {1}\n'.format(self.name, time))
        timeDir = '{0}/{1}/'.format(self.mesh.case, time)
        if not exists(timeDir):
            makedirs(timeDir)
        handle = open(timeDir + self.name, 'w')
        handle.write(utils.foamHeader)
        handle.write('FoamFile\n{\n')
        foamFile = utils.foamFile.copy()
        foamFile['object'] = self.name
        for key in foamFile:
            handle.write('\t' + key + ' ' + foamFile[key] + ';\n')
        handle.write('}\n')
        handle.write('// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n')
        handle.write('dimensions      [0 1 -1 0 0 0 0];\n')
        handle.write('internalField   nonuniform List<scalar>\n')
        handle.write('{0}\n(\n'.format(self.mesh.nInternalCells))
        for value in ad.value(self.getInternalField()):
            handle.write(str(value[0]) + '\n')
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
            patch = mesh.boundary[patchID] 
            startFace = patch['startFace']
            nFaces = patch['nFaces']
            endFace = startFace + nFaces
            indices = mesh.nInternalCells + range(startFace, endFace) - mesh.nInternalFaces 
            if patch['type'] == 'cyclic':
                neighbourPatch = mesh.boundary[patch['neighbourPatch']]   
                neighbourStartFace = neighbourPatch['startFace']
                neighbourEndFace = neighbourStartFace + nFaces
                self.field[indices] = self.field[mesh.owner[neighbourStartFace:neighbourEndFace]]
            else:
                boundaryCondition = self.boundary[patchID]['type']
                getattr(BCs, boundaryCondition)(self, indices, np.arange(startFace, endFace))


