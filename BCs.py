import numpy as np

import config
from config import ad, adsparse, T
from mesh import extractField
logger = config.Logger(__name__)

# from charles
# WALL BC's
# isothermal: p_wall extrapolated, u, T set
# adiabatic : p_wall extrapolated, u set, dt/dn set
# CBC
# internal extrapolated using gradient
# external set
# calcEulerFlux applied

def dot(v, w, dims=1):
    if dims == 1:
        return ad.sum(v*w, axis=1, keepdims=True)
    else:
        w = w[:,np.newaxis,:]
        return ad.sum(v*w, axis=-1)
def cross(v, w):
    p = v*0.
    p[:,0] = v[:,1]*w[:,2]-v[:,2]*w[:,1]
    p[:,1] = v[:,2]*w[:,0]-v[:,0]*w[:,2]
    p[:,2] = v[:,0]*w[:,1]-v[:,1]*w[:,0]
    return p

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
        self.normals = self.mesh.normals[self.startFace:self.endFace]
        # used by field writer
        self.getValue = lambda: self.field[self.cellStartFace:self.cellEndFace]
        self.inputs = []

    def createInput(self, key, dimensions):
        nFaces = self.mesh.origMesh.boundary[self.patchID]['nFaces']
        value = extractField(self.patch[key], nFaces, dimensions)
        symbolic = ad.matrix()
        self.inputs.append((symbolic, value))
        return symbolic

    def setValue(self, value):
        self.phi.field = ad.set_subtensor(self.phi.field[self.cellStartFace:self.cellEndFace], value)

    def update(self):
        pass

class calculated(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        if 'value' in self.patch:
            self.fixedValue = self.createInput('value', self.phi.dimensions)
            self.setValue(self.fixedValue)
        else:
            self.setValue(0.)

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
        self.setValue(self.phi.field[self.neighbourIndices])

class slidingPeriodic1D(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        neighbourPatch = self.mesh.boundary[patchID]['neighbourPatch']
        neighbourStartFace = self.mesh.boundary[neighbourPatch]['startFace']
        neighbourEndFace = neighbourStartFace + self.nFaces
        self.neighbourIndices = self.mesh.owner[neighbourStartFace:neighbourEndFace]
        self.interpOp = self.mesh.boundary[patchID]['loc_multiplier']
        self.velocity = self.mesh.origMesh.boundary[patchID]['loc_velocity']

    def update(self):
        logger.debug('slidingPeriodic1D BC for {0}'.format(self.patchID))
        neighbourPhi = self.phi.field[self.neighbourIndices]
        if len(self.phi.dimensions) == 2:
            neighbourPhi = neighbourPhi.reshape((self.nFaces, 9))
        value = adsparse.basic.dot(self.interpOp, neighbourPhi)
        if len(self.phi.dimensions) == 2:
            value = value.reshape((self.nFaces, 3, 3))
        if self.phi.name == 'U':
            value = value + self.velocity
        self.setValue(value)

class zeroGradient(BoundaryCondition):
    def update(self):
        logger.debug('zeroGradient BC for {0}'.format(self.patchID))
        boundaryValue = self.phi.field[self.internalIndices]
        if hasattr(self.phi, 'grad'):
            # second order correction
            grad = self.phi.grad.field[self.internalIndices]
            R = self.mesh.faceCentres[self.startFace:self.endFace] - self.mesh.cellCentres[self.internalIndices]
            #boundaryValue = boundaryValue + 0.5*dot(grad, R, self.phi.dimensions[0])
        self.setValue(boundaryValue)

class symmetryPlane(zeroGradient):
    def update(self):
        logger.debug('symmetryPlane BC for {0}'.format(self.patchID))
        super(self.__class__, self).update()
        # if vector
        if self.phi.dimensions == (3,):
            v = -self.normals
            value = self.phi.field[self.cellStartFace:self.cellEndFace]
            self.setValue(value-dot(value, v)*v)

class fixedValue(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        # mesh values required outside theano
        #self.fixedValue = extractField(self.patch['value'], self.nFaces, self.phi.dimensions == (3,))
        self.fixedValue = self.createInput('value', self.phi.dimensions)

    def update(self):
        logger.debug('fixedValue BC for {0}'.format(self.patchID))
        self.setValue(self.fixedValue)

class CharacteristicBoundaryCondition(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(CharacteristicBoundaryCondition, self).__init__(phi, patchID)
        self.U, self.T, _ = self.solver.getBCFields()
        self.p = self.phi

class CBC_UPT(CharacteristicBoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.mesh.boundary[patchID]['type'] = 'characteristic'
        self.U0 = self.createInput('U0', (3,))
        self.T0 = self.createInput('T0', (1,))
        self.p0 = self.createInput('p0', (1,))

    def update(self):
        self.U.field = ad.set_subtensor(self.U.field[self.cellStartFace:self.cellEndFace], self.U0)
        self.T.field = ad.set_subtensor(self.T.field[self.cellStartFace:self.cellEndFace], self.T0)
        self.p.field = ad.set_subtensor(self.p.field[self.cellStartFace:self.cellEndFace], self.p0)

# implement support for characteristic time travel
class CBC_TOTAL_PT(CharacteristicBoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.mesh.boundary[patchID]['type'] = 'characteristic'
        self.Tt = self.createInput('Tt', (1,))
        self.pt = self.createInput('pt', (1,))
        self.Cp = self.solver.Cp
        self.gamma = self.solver.gamma

    def update(self):
        Un = dot(self.U.field[self.internalIndices], self.normals)
        U = Un*self.normals
        T = self.Tt - 0.5*Un*Un/self.Cp
        p = self.pt * (T/self.Tt)**(self.gamma/(self.gamma-1))
        
        self.U.field = ad.set_subtensor(self.U.field[self.cellStartFace:self.cellEndFace], U)
        self.T.field = ad.set_subtensor(self.T.field[self.cellStartFace:self.cellEndFace], T)
        self.p.field = ad.set_subtensor(self.p.field[self.cellStartFace:self.cellEndFace], p)

class nonReflectingOutletPressure(CharacteristicBoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        nFaces = self.mesh.origMesh.boundary[patchID]['nFaces']
        self.Ma = float(self.patch['Mamax'])
        self.L = float(self.patch['L'])
        self.pt = self.createInput('ptar', (1,))
        self.gamma = self.solver.gamma
        self.value = self.createInput('value', self.phi.dimensions)

        self.UBC = zeroGradient(self.U, patchID)
        self.TBC = zeroGradient(self.T, patchID)

    def update(self):
        self.UBC.update()
        self.TBC.update()
        if self.solver.stage == 0:
            self.setValue(self.value)
        else:
            # FORWARD EULER INTEGRATION
            if self.solver.stage == 1:
                #self.p0 = self.phi.field[self.cellStartFace:self.cellEndFace]
                self.p0 = self.phi.field[self.internalIndices]
                self.c = self.solver.aF.field[self.startFace:self.endFace]
                self.Un0 = dot(self.U.field[self.internalIndices], self.normals)
            if self.solver.stage == self.solver.nStages:
                Un = dot(self.U.field[self.internalIndices], self.normals)
                rho = self.gamma*self.p0/(self.c*self.c)
                dt = self.solver.dt
                if self.solver.localTimeStep:
                    dt = dt[self.internalIndices]
                p = self.p0
                p = p + rho*self.c*(Un-self.Un0)
                p = p + dt*(-0.25*self.c*(1-self.Ma*self.Ma)*(self.p0-self.pt)/self.L)
                self.setValue(p)
            else:
                self.setValue(self.p0)

class turbulentInletVelocity(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(self.__class__, self).__init__(phi, patchID)
        self.Umean = self.createInput('Umean', (3,))
        self.lengthScale = float(self.patch['lengthScale'])
        self.timeScale = float(self.patch['timeScale'])
        r11 = float(self.patch['r11'])
        r12 = float(self.patch['r12'])
        r13 = float(self.patch['r13'])
        r22 = float(self.patch['r22'])
        r23 = float(self.patch['r23'])
        r33 = float(self.patch['r33'])
        R = np.array([[r11,0,0],[r12,r22,0],[r13,r23,r33]])
        w, v = np.linalg.eigh(R)
        self.c = np.sqrt(w)
        self.T = v

        #self.patch.pop('value', None)
        #self.patch['value'] = 'uniform (0 0 0)'
        self.N = 100
        self.omega = np.random.randn(self.N, 1)
        self.kd = np.random.randn(self.N, 3)/2**0.5
        self.psi = np.random.rand(self.N, 3)

        self.Uscale = self.lengthScale/self.timeScale
        self.k = self.kd*self.Uscale/self.c

    def update(self):
        logger.debug('turbulentInletVelocity BC for {0}'.format(self.patchID))
        #self.value[:] = self.fixedValue
        value = self.Umean
        if self.solver.stage == 0:
            self.setValue(self.Umean)
        else:
            x = self.mesh.cellCentres[self.cellStartFace:self.cellEndFace]
            x = ad.ifelse(ad.eq(x.shape[0], 0), x, x-x[0,:])
            x = x / self.lengthScale
            t = self.solver.t/self.timeScale
            p = cross(self.kd, self.psi)[:,np.newaxis,:]
            phi = ad.cos((self.k[:,np.newaxis,:]*x[np.newaxis,:,:]).sum(axis=2) + self.omega*t)[:,:,np.newaxis]
            value = value + self.c*(2./self.N)**0.5*(p*phi).sum(axis=0)
            value = ad.sum(self.T.T*value[:,:,np.newaxis], axis=1)
            self.setValue(value)

slip = symmetryPlane
empty = zeroGradient
inletOutlet = zeroGradient
   
#class totalPressure(BoundaryCondition):
#    def __init__(self, phi, patchID):
#        super(self.__class__, self).__init__(phi, patchID)
#        self.p0 = extractFeld(self.patch['p0'], self.nFaces, (1,))
#        self.gamma = self.patch['gamma']
#        self.patch.pop('value', None)
#        self.patch['value'] = 'uniform (0 0 0)'
#
#    def update(self):
#        logger.debug('totalPressure BC for {0}'.format(self.patchID))
#        #self.value[:] = self.fixedValue
#        T = self.solver.T.field[self.internalIndices]
#        U = self.solver.U.field[self.internalIndices]
#        R = solver.R
#        M = 0.0289
#        c = (self.gamma*R*T/M)**0.5
#        Ma = utils.norm(U, axis=1)/c
#        p = p0/(1+(self.gamma-1)*0.5*Ma**2)**(self.gamma/(self.gamma-1))
#        self.field[self.cellStartFace:self.cellEndFace] = p
#
#class pressureInletOutletVelocity(BoundaryCondition):
#    def update(self):
#        logger.debug('pressureInletOutlletVelocity BC for {0}'.format(self.patchID))
#        phi = self.solver.flux.field[self.startFace, self.endFace]
#        n = self.mesh.normals[self.startFace, self.endFace]
#        self.field[self.cellStartFace:self.cellEndFace] = phi*n

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


