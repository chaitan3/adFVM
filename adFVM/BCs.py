import numpy as np

from . import config
from .tensor import Tensor, Kernel, StaticVariable
from .mesh import extractField, extractVector
logger = config.Logger(__name__)

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


def dot(v, w, dims=1):
    if dims == 1:
        return ad.sum(v*w, axis=1, keep_dims=True)
    else:
        w = w[:,np.newaxis,:]
        return ad.sum(v*w, axis=-1)
def cross(v, w):
    p0 = v[:,1]*w[:,2]-v[:,2]*w[:,1]
    p1 = v[:,2]*w[:,0]-v[:,0]*w[:,2]
    p2 = v[:,0]*w[:,1]-v[:,1]*w[:,0]
    p = tf.stack((p0,p1,p2), axis=1)
    return p

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
        # mesh values required outside theano
        #self.fixedValue = extractField(self.patch['value'], self.nFaces, self.phi.dimensions == (3,))
        self.createInput('value', self.dimensions)

    def _update(self, phi, fixedValue):
        logger.debug('fixedValue BC for {0}'.format(self.patchID))
        return fixedValue*1

    
class CharacteristicBoundaryCondition(BoundaryCondition):
    def __init__(self, phi, patchID):
        super(CharacteristicBoundaryCondition, self).__init__(phi, patchID)
        #self.U, self.T, _ = self.solver.getBCFields()
        #self.p = self.phi

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
        return U0, T0, p0

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

#class symmetryPlane(zeroGradient):
#    def _update(self):
#        logger.debug('symmetryPlane BC for {0}'.format(self.patchID))
#        dims = self.phi.dimensions
#        phi = Tensor(dims)
#        # if vector
#        if dims == (3,):
#            v = -self.normals
#            #value = self.phi.field[self.cellStartFace:self.cellEndFace]
#            phiF = phi-phi.dot(v)*v
#        else:
#            phiF = phi*1
#        return TensorFunction('symmetryPlane', [phi, self.normals], [phiF])
#
#slip = symmetryPlane

#class slidingPeriodic1D(BoundaryCondition):
#    def __init__(self, phi, patchID):
#        super(self.__class__, self).__init__(phi, patchID)
#        neighbourPatch = self.mesh.boundary[patchID]['neighbourPatch']
#        neighbourStartFace = self.mesh.boundary[neighbourPatch]['startFace']
#        neighbourEndFace = neighbourStartFace + self.nFaces
#        self.neighbourIndices = self.mesh.owner[neighbourStartFace:neighbourEndFace]
#        self.interpOp = self.mesh.boundary[patchID]['loc_multiplier']
#        self.velocity = self.mesh.origMesh.boundary[patchID]['loc_velocity']
#
#    def _update(self):
#        logger.debug('slidingPeriodic1D BC for {0}'.format(self.patchID))
#        #m = self.mesh.origMesh
#        #patch = m.boundary[self.patchID]
#        #cellStartFace = patch['startFace']  + m.nInternalCells-m.nInternalFaces
#        #cellEndFace  = cellStartFace + patch['nFaces']
#        #print(cellStartFace, cellEndFace, self.phi.name, self.patchID)
#        neighbourPhi = ad.gather(self.phi.field, self.neighbourIndices)
#        if len(self.phi.dimensions) == 2:
#            neighbourPhi = neighbourPhi.reshape((self.nFaces, 9))
#        value = adsparse.basic.dot(self.interpOp, neighbourPhi)
#        if len(self.phi.dimensions) == 2:
#            value = value.reshape((self.nFaces, 3, 3))
#        if self.phi.name == 'U':
#            value = value + self.velocity
#        self.setValue(value)


#
#class nonReflectingOutletPressure(CharacteristicBoundaryCondition):
#    def __init__(self, phi, patchID):
#        super(self.__class__, self).__init__(phi, patchID)
#        self.Ma = float(self.patch['Mamax'])
#        self.L = float(self.patch['L'])
#        self.pt = self.createInput('ptar', (1,))
#        self.gamma = self.solver.gamma
#        self.value = self.createInput('value', self.phi.dimensions)
#
#        self.UBC = zeroGradient(self.U, patchID)
#        self.TBC = zeroGradient(self.T, patchID)
#
#    def _update(self):
#        raise Exception('TODO')
#        self.UBC.update()
#        self.TBC.update()
#        if self.solver.stage == 0:
#            self.setValue(self.value)
#        else:
#            alpha, beta, _ = self.solver.timeStepCoeff
#            i = self.solver.stage - 1
#
#            dt = self.solver.dt
#            if self.solver.localTimeStep:
#                dt = dt[self.internalIndices]
#
#            c = self.solver.aF.field[self.startFace:self.endFace]
#            if i == 0:
#                self.p = [self.phi.field[self.cellStartFace:self.cellEndFace]]
#                self.dp = []
#                Un = dot(self.U.field[self.internalIndices], self.normals)
#                Un0 = dot(self.U.field[self.cellStartFace:self.cellEndFace], self.normals)
#                rho = self.gamma*self.p[i]/(c*c)
#                self.dUn = rho*c*(Un-Un0)
#            dp = -0.25*c*(1-self.Ma**2)*(self.p[i]-self.pt)/self.L
#            self.dp.append(dp)
#            p = self.dUn
#            for j in range(0, i+1):
#                p += alpha[i,j]*self.p[j] + beta[i,j]*self.dp[j]*dt
#            self.p.append(p)
#            self.setValue(p) 
#
#            # FORWARD EULER INTEGRATION
#            #if self.solver.stage == 1:
#            #    self.p0 = self.phi.field[self.cellStartFace:self.cellEndFace]
#            #    #self.p0 = self.phi.field[self.internalIndices]
#            #    self.c = self.solver.aF.field[self.startFace:self.endFace]
#            #    self.Un0 = dot(self.U.field[self.internalIndices], self.normals)
#            #if self.solver.stage == self.solver.nStages:
#            #    Un = dot(self.U.field[self.internalIndices], self.normals)
#            #    rho = self.gamma*self.p0/(self.c*self.c)
#            #    dt = self.solver.dt
#            #    if self.solver.localTimeStep:
#            #        dt = dt[self.internalIndices]
#            #    p = self.p0
#            #    p = p + rho*self.c*(Un-self.Un0)
#            #    p = p + dt*(-0.25*self.c*(1-self.Ma*self.Ma)*(self.p0-self.pt)/self.L)
#            #    self.setValue(p)
#            #else:
#            #    self.setValue(self.p0)
#
## adjoint?
#class turbulentInletVelocityN(BoundaryCondition):
#    def __init__(self, phi, patchID):
#        super(self.__class__, self).__init__(phi, patchID)
#        self.Umean = self.createInput('Umean', (3,))
#        self.value = self.createInput('value', self.phi.dimensions)
#        self.lengthScale = float(self.patch['lengthScale'])
#        self.timeScale = float(self.patch['timeScale'])
#        r11 = float(self.patch['r11'])
#        r12 = float(self.patch['r12'])
#        r13 = float(self.patch['r13'])
#        r22 = float(self.patch['r22'])
#        r23 = float(self.patch['r23'])
#        r33 = float(self.patch['r33'])
#        self.x0 = np.array(extractVector(self.patch['x0'])).astype(config.precision)
#        R = np.array([[r11,0,0],[r12,r22,0],[r13,r23,r33]]).astype(config.precision)
#        w, v = np.linalg.eigh(R)
#        self.c = np.sqrt(w)
#        self.T = v
#
#        #self.patch.pop('value', None)
#        #self.patch['value'] = 'uniform (0 0 0)'
#        self.N = 100
#        self.omega = np.random.randn(self.N, 1)
#        self.kd = np.random.randn(self.N, 3)/2**0.5
#        self.psi = np.random.rand(self.N, 3)
#
#        self.Uscale = self.lengthScale/self.timeScale
#        self.k = self.kd#*self.Uscale/self.c
#
#    def _update(self):
#        logger.debug('turbulentInletVelocity BC for {0}'.format(self.patchID))
#        #self.value[:] = self.fixedValue
#        value = self.value
#        if self.solver.stage > 0:
#            x = self.mesh.cellCentres[self.cellStartFace:self.cellEndFace]
#            x = (x-self.x0) / self.lengthScale
#            t = self.solver.t/self.timeScale
#            p = cross(self.kd, self.psi)[:,np.newaxis,:]
#            phi = ad.cos((self.k[:,np.newaxis,:]*x[np.newaxis,:,:]).sum(axis=2) + self.omega*t)[:,:,np.newaxis]
#            value = value + self.c*(2./self.N)**0.5*(p*phi).sum(axis=0)
#            #pprint('HACK')
#            #f = (1./0.265)*ad.exp(-(1/(1.00001-(self.mesh.cellCentres[self.cellStartFace:self.cellEndFace,[1]]-0.5)**2)))
#            #value = value + self.c*f*(2./self.N)**0.5*(p*phi).sum(axis=0)
#            value = ad.sum(self.T.T*value[:,:,np.newaxis], axis=1)
#        self.setValue(value)
#
#class turbulentInletVelocity(BoundaryCondition):
#    def __init__(self, phi, patchID):
#        super(self.__class__, self).__init__(phi, patchID)
#        self.Umean = self.createInput('Umean', (3,))
#
#    def _update(self):
#        self.setValue(self.Umean)


   
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


