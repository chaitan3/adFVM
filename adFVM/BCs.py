import numpy as np

from . import config
from .mesh import extractField, extractVector
logger = config.Logger(__name__)

from adpy.tensor import Tensor, Kernel, StaticVariable

# from charles
# WALL BC's
# isothermal: p_wall extrapolated, u, T set
# adiabatic : p_wall extrapolated, u set, dt/dn set
# CBC
# internal extrapolated using gradient
# external set
# calcEulerFlux applied

valuePatches = config.processorPatches + ['calculated',
                                          'CBC_UPT',
                                          'CBC_TOTAL_PT',
                                          'nonReflectingOutletPressure',
                                          'turbulentInletVelocity'
                                          ]


class BoundaryCondition(object):
    def __init__(self, phi, patchID):
        self.patchID = patchID
        self.dimensions = phi.dimensions
        self.solver = phi.solver
        self.mesh = phi.mesh
        self.patch = phi.boundary[patchID]

        mesh = self.mesh.symMesh
        patch = mesh.boundary[patchID]
        self.startFace, self.nFaces = patch['startFace'], patch['nFaces']
        self.cellStartFace = patch['cellStartFace']
        self.normals = mesh.normals[self.startFace]
        self.owner = mesh.owner
        # used by field writer
        self.keys = []
        self.inputs = []
        self._tensorUpdate = Kernel(self._update)

    def createInput(self, key, dimensions):
        patch = self.mesh.boundary[self.patchID]
        nFaces = patch['nFaces']
        value = extractField(self.patch[key], nFaces, dimensions)
        self.patch['_{}'.format(key)] = value
        symbolic = StaticVariable((self.nFaces,) + dimensions)
        #print(symbolic.dtype)
        self.keys.append(key)
        self.inputs.append((symbolic, value))
        return symbolic

    def getInputs(self):
        return [x for x in self.inputs if x[1] is not None]

    def _update(self):
        pass

    def update(self, phi):
        inputs = tuple([phi] + [x[0] for x in self.inputs])
        outputs = (phi[self.cellStartFace],)
        phi = self._tensorUpdate(self.nFaces, outputs)(*inputs)[0]
        patch = self.mesh.boundary[self.patchID]
        info = [self.__class__, self.patchID, patch['startFace'], patch['nFaces'], patch['cellStartFace']]
        patch = self.mesh.symMesh.boundary[self.patchID]
        info += [patch['startFace'].name, patch['nFaces'].name, patch['cellStartFace'].name]
        #print(info)
        phi.args[0].info += info
        return phi

class calculated(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        #if 'value' in self.patch:
        #    self.fixedValue = self.createInput('value', self.phi.dimensions)
        #    self.setValue(self.fixedValue)
        #else:
        #    self.setValue(0.)

    def update(self, phi):
        return phi

class cyclic(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        neighbourPatch = self.mesh.boundary[patchID]['neighbourPatch']
        neighbourStartFace = self.mesh.symMesh.boundary[neighbourPatch]['startFace']
        self.inputs.append((self.owner[neighbourStartFace], None))

    def _update(self, phi, neighbourIndices):
        logger.debug('cyclic BC for {0}'.format(self.patchID))
        return phi.extract(neighbourIndices)

class zeroGradient(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.inputs.append((self.owner[self.startFace], None))

    def _update(self, phi, owner):
        logger.debug('zeroGradient BC for {0}'.format(self.patchID))
        return phi.extract(owner)
        #if hasattr(self.phi, 'grad'):
        #    # second order correction
        #    grad = self.phi.grad.field[self.internalIndices]
        #    R = self.mesh.faceCentres[self.startFace:self.endFace] - self.mesh.cellCentres[self.internalIndices]
        #    boundaryValue = boundaryValue + 0.5*dot(grad, R, self.phi.dimensions[0])

empty = zeroGradient
inletOutlet = zeroGradient

class fixedValue(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.createInput('value', self.dimensions)

    def _update(self, phi, fixedValue):
        logger.debug('fixedValue BC for {0}'.format(self.patchID))
        return fixedValue*1

    
class CharacteristicBoundaryCondition(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(CharacteristicBoundaryCondition, self).__init__(phi, patchID)

    def update(self, *args):
        if len(args) == 1:
            return args[0]
        U, T, p = args
        inputs = tuple([U, T, p] + [x[0] for x in self.inputs])
        outputs = (U[self.cellStartFace],T[self.cellStartFace], p[self.cellStartFace])
        return Kernel(self._update)(self.nFaces, outputs)(*inputs)

class CBC_UPT(CharacteristicBoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.mesh.boundary[patchID]['type'] = 'characteristic'
        self.createInput('U0', (3,))
        self.createInput('T0', (1,))
        self.createInput('p0', (1,))

    def _update(self, U, T, p, U0, T0, p0):
        return U0*1, T0*1, p0*1

# implement support for characteristic time travel
class CBC_TOTAL_PT(CharacteristicBoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.mesh.boundary[patchID]['type'] = 'characteristic'
        self.createInput('Tt', (1,))
        self.createInput('pt', (1,))
        self.Cp = self.solver.Cp
        self.gamma = self.solver.gamma
        if 'direction' in self.patch:
            self.createInput('direction', (3,))
        else:
            self.inputs.append((self.normals, None))
        self.inputs.append((self.owner[self.startFace], None))

    def _update(self, U, T, p, Tt, pt, direction, owner):
        U, T = U.extract(owner), T.extract(owner)
        Un = U.dot(direction)
        Ub = Un*direction
        Tb = Tt - 0.5*Un*Un/self.Cp
        pb = pt * (T/Tt)**(self.gamma/(self.gamma-1))
        return Ub, Tb, pb

