from __future__ import print_function
import numpy as np

from utils import ad
from utils import Logger
logger = Logger(__name__)
import utils

class BoundaryCondition(object):
    def __init__(self, phi, patchID):
        self.patch = phi.boundary[patchID]
        self.patchID = patchID
        self.mesh = phi.mesh
        self.field = phi.field
        self.startFace = self.mesh.boundary[patchID]['startFace']
        self.nFaces = self.mesh.boundary[patchID]['nFaces']
        self.endFace = self.startFace + self.nFaces
        self.cellStartFace = self.mesh.nInternalCells + self.startFace - self.mesh.nInternalFaces
        self.cellEndFace = self.mesh.nInternalCells + self.endFace - self.mesh.nInternalFaces
        self.internalIndices = self.mesh.owner[self.startFace:self.endFace]
        self.value = self.field[self.cellStartFace:self.cellEndFace]

class calculated(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        if 'value' in self.patch:
            self.field[self.cellStartFace:self.cellEndFace] = utils.extractField(self.patch['value'], self.nFaces, self.field.shape == 3)
        else:
            self.field[self.cellStartFace:self.cellEndFace] = 0.

    def update(self):
        pass

class cyclic(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        neighbourPatch = self.mesh.boundary[patchID]['neighbourPatch']
        neighbourStartFace = self.mesh.boundary[neighbourPatch]['startFace']
        neighbourEndFace = neighbourStartFace + self.nFaces
        self.neighbourIndices = self.mesh.owner[neighbourStartFace:neighbourEndFace]

    def update(self):
        logger.debug('cyclic BC for {0}'.format(self.patchID))
        #self.value[:] = self.field[self.neighbourIndices]
        self.field[self.cellStartFace:self.cellEndFace] = self.field[self.neighbourIndices]

class processor(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.local = self.mesh.boundary[patchID]['myProcNo']
        self.remote = self.mesh.boundary[patchID]['neighbProcNo']

    def update(self, exchanger):
        logger.debug('processor BC for {0}'.format(self.patchID))
        exchanger.exchange(self.remote, self.field[self.internalIndices], self.value)

class zeroGradient(BoundaryCondition):
    def update(self):
        logger.debug('zeroGradient BC for {0}'.format(self.patchID))
        #self.value[:] = self.field[self.internalIndices]
        self.field[self.cellStartFace:self.cellEndFace] = self.field[self.internalIndices]

class symmetryPlane(zeroGradient):
    def update(self):
        logger.debug('symmetryPlane BC for {0}'.format(self.patchID))
        super(self.__class__, self).update()
        # if vector
        if self.field.shape[1:] == (3,):
            v = -self.mesh.normals[self.startFace:self.endFace]
            #self.value -= ad.sum(self.value*v, axis=1).reshape((-1,1))*v
            self.field[self.cellStartFace:self.cellEndFace] -= ad.sum(self.field[self.cellStartFace:self.cellEndFace]*v, axis=1).reshape((-1,1))*v

class fixedValue(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.fixedValue = utils.extractField(self.patch['value'], self.nFaces, self.field.shape[1:] == (3,))

    def update(self):
        logger.debug('fixedValue BC for {0}'.format(self.patchID))
        #self.value[:] = self.fixedValue
        self.field[self.cellStartFace:self.cellEndFace] = self.fixedValue

slip = symmetryPlane
empty = zeroGradient
inletOutlet = zeroGradient
    

