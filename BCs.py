import numpy as np

import config
from config import ad, T
from mesh import extractField
logger = config.Logger(__name__)

class BoundaryCondition(object):
    def __init__(self, phi, patchID):
        self.patchID = patchID
        self.phi = phi
        self.solver = phi.solver
        self.mesh = phi.mesh
        self.patch = phi.boundary[patchID]
        self.startFace = self.mesh.boundary[patchID]['startFace']
        self.nFaces = self.mesh.boundary[patchID]['nFaces']
        self.endFace = self.startFace + self.nFaces
        self.cellStartFace = self.mesh.nInternalCells + self.startFace - self.mesh.nInternalFaces
        self.cellEndFace = self.mesh.nInternalCells + self.endFace - self.mesh.nInternalFaces
        self.internalIndices = self.mesh.owner[self.startFace:self.endFace]
        # used by field writer
        self.getValue = lambda: self.field[self.cellStartFace:self.cellEndFace]
        # used by processor patches
        #self.value = self.field[self.cellStartFace:self.cellEndFace]

class calculated(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        if 'value' in self.patch:
            self.phi.field = ad.set_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], extractField(self.patch['value'], self.nFaces, self.field.shape[1:]))
        else:
            self.phi.field = ad.set_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], 0)

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
        self.phi.field = ad.set_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], self.phi.field[self.neighbourIndices])

#class processor(BoundaryCondition):
#    def __init__(self, phi, patchID):
#        super(processor, self).__init__(phi, patchID)
#        self.local = self.mesh.boundary[patchID]['myProcNo']
#        self.remote = self.mesh.boundary[patchID]['neighbProcNo']
#        self.patch.pop('value', None)
#        self.tag = 0
#
#    def update(self, exchanger):
#        logger.debug('processor BC for {0}'.format(self.patchID))
#        exchanger.exchange(self.remote, self.field[self.internalIndices], self.value, self.tag)
#
#class processorCyclic(processor):
#    def __init__(self, phi, patchID):
#        super(processorCyclic, self).__init__(phi, patchID)
#        commonPatch = self.mesh.boundary[patchID]['referPatch']
#        if self.local > self.remote:
#            commonPatch = self.mesh.boundary[commonPatch]['neighbourPatch']
#        self.tag = 1 + self.mesh.origPatches.index(commonPatch)

class zeroGradient(BoundaryCondition):
    def update(self):
        logger.debug('zeroGradient BC for {0}'.format(self.patchID))
        #self.value[:] = self.field[self.internalIndices]
        self.phi.field = ad.set_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], self.phi.field[self.internalIndices])

class symmetryPlane(zeroGradient):
    def update(self):
        logger.debug('symmetryPlane BC for {0}'.format(self.patchID))
        super(self.__class__, self).update()
        # if vector
        if self.phi.dimensions == (3,):
            v = -self.mesh.normals[self.startFace:self.endFace]
            self.phi.field = ad.inc_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], -ad.sum(self.phi.field[self.cellStartFace:self.cellEndFace]*v, axis=1).reshape((self.nFaces,1))*v)

class fixedValue(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        # mesh values required outside theano
        #self.fixedValue = extractField(self.patch['value'], self.nFaces, self.phi.dimensions == (3,))
        self.fixedValue = extractField(self.patch['value'], self.mesh.origMesh.boundary[patchID]['nFaces'], self.phi.dimensions)
        self.fixedValue = T.shared(self.fixedValue)

    def update(self):
        logger.debug('fixedValue BC for {0}'.format(self.patchID))
        self.phi.field = ad.set_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], self.fixedValue)

class turbulentInletVelocity(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.Umean = extractField(self.patch['Umean'], self.nFaces, self.field.shape[1:])
        self.lengthScale = self.patch['lengthScale']
        self.turbulentIntensity = self.patch['turbulentIntensity']
        self.patch.pop('value', None)
        self.patch['value'] = 'uniform (0 0 0)'

    def update(self):
        logger.debug('turbulentInletVelocity BC for {0}'.format(self.patchID))
        #self.value[:] = self.fixedValue
        self.field[self.cellStartFace:self.cellEndFace] = self.Umean

class totalPressure(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.p0 = extractField(self.patch['p0'], self.nFaces, (1,))
        self.gamma = self.patch['gamma']
        self.patch.pop('value', None)
        self.patch['value'] = 'uniform (0 0 0)'

    def update(self):
        logger.debug('totalPressure BC for {0}'.format(self.patchID))
        #self.value[:] = self.fixedValue
        T = self.solver.T.field[self.internalIndices]
        U = self.solver.U.field[self.internalIndices]
        R = solver.R
        M = 0.0289
        c = (self.gamma*R*T/M)**0.5
        Ma = utils.norm(U, axis=1)/c
        p = p0/(1+(self.gamma-1)*0.5*Ma**2)**(self.gamma/(self.gamma-1))
        self.field[self.cellStartFace:self.cellEndFace] = p

class pressureInletOutletVelocity(BoundaryCondition):
    def update(self):
        logger.debug('pressureInletOutlletVelocity BC for {0}'.format(self.patchID))
        phi = self.solver.flux.field[self.startFace, self.endFace]
        n = self.mesh.normals[self.startFace, self.endFace]
        self.field[self.cellStartFace:self.cellEndFace] = phi*n

slip = symmetryPlane
empty = zeroGradient
inletOutlet = zeroGradient
    

