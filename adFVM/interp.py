from . import config, compat, parallel
from .config import ad, T
from field import Field, faceExchange

import itertools
import numpy as np

logger = config.Logger(__name__)

# central is second order with no grad
# dual should do second order with grad
# characteristic is extrapolated

# dual defined for np and ad
def central(phi, mesh):
    logger.info('interpolating {0}'.format(phi.name))
    factor = mesh.weights
    # for tensor
    if len(phi.dimensions) == 2:
        factor = factor.reshape((factor.shape[0], 1, 1))
    faceField = Field('{0}F'.format(phi.name), phi.field[mesh.owner]*factor + phi.field[mesh.neighbour]*(1.-factor), phi.dimensions)
    #faceField = Field('{0}F'.format(phi.name), phi.field[mesh.owner] + phi.field[mesh.neighbour], phi.dimensions)
    # retain pattern broadcasting
    if hasattr(mesh, 'origMesh'):
        faceField.field = ad.patternbroadcast(faceField.field, phi.field.broadcastable)
    return faceField

class Reconstruct(object):
    def __init__(self, solver):
        self.solver = solver
        self.mesh = solver.mesh
        mesh = self.mesh
        meshO = mesh.origMesh
        indices = [np.arange(0, meshO.nInternalFaces)]
        Cindices = []
        Bindices = []
        for patchID in mesh.localPatches:
            startFace, endFace, _ = meshO.getPatchFaceRange(patchID)
            patchType = mesh.boundary[patchID]['type']
            patchIndices = np.arange(startFace, endFace)
            if patchType in config.cyclicPatches:
                indices.append(patchIndices)
            elif patchType == 'characteristic':
                Cindices.append(patchIndices)
            else:
                Bindices.append(patchIndices)

        nRemoteFaces = meshO.nFaces-(meshO.nCells-meshO.nLocalCells)
        indices.append(np.arange(nRemoteFaces, meshO.nFaces))
        indices = np.concatenate(indices).astype(np.int32)
        self.indices = ad.ivector()
        solver.postpro.append((self.indices, indices))

        self.boundary = len(Bindices) > 0
        if self.boundary:
            Bindices = np.concatenate(Bindices).astype(np.int32)
            self.Bindices = ad.ivector()
            solver.postpro.append((self.Bindices, Bindices))

        self.characteristic = len(Cindices) > 0
        if self.characteristic:
            Cindices = np.concatenate(Cindices).astype(np.int32)
            self.Cindices = ad.ivector()
            solver.postpro.append((self.Cindices, Cindices))

        self.valueIndices = [indices, Bindices, Cindices]
        owner, neighbour = mesh.owner[self.indices], mesh.neighbour[self.indices]
        self.faceOptions = [[owner, neighbour], [neighbour, owner]]

        self.cyclicStartFaces = {}
        startFace = self.mesh.nInternalFaces.copy()
        for patchID in self.mesh.localPatches:
            patch = self.mesh.boundary[patchID]
            if patch['type'] in config.cyclicPatches:
                self.cyclicStartFaces[patchID] = startFace
                startFace += patch['nFaces']
        self.procStartFace = startFace
        self.BCUpdate = False

        return

    def dual(self, phi, gradPhi):
        assert len(phi.dimensions) == 1
        logger.info('TVD {0}'.format(phi.name))

        faceFields = []
        faceFields.append(self.update(0, phi, gradPhi))
        faceFields.append(self.update(1, phi, gradPhi))

        faceFieldsF = [Field('{0}F'.format(phi.name), faceField, phi.dimensions) for faceField in faceFields]
        if self.BCUpdate:
            self.faceBCUpdate(faceFieldsF)
        return faceFieldsF

    def dualSystem(self, *fieldsSys):
        phiFSys = []
        for phi, gradPhi in fieldsSys:
            phiF = self.dual(phi, gradPhi)
            phiFSys.append(phiF)
        return phiFSys

    def faceBCUpdate(self, faceFieldsF):
        faceFields = [phi.field for phi in faceFieldsF]
        faceFields[1] = ad.concatenate((faceFields[1], faceFields[0][self.mesh.nInternalFaces:]), axis=0)
        for patchID in self.mesh.localPatches:
            patch = self.mesh.boundary[patchID]
            if patch['type'] in config.cyclicPatches:
                startFace = self.cyclicStartFaces[patchID]
                nFaces = patch['nFaces']
                neighbourStartFace = self.cyclicStartFaces[patch['neighbourPatch']]
                faceFieldsF[1].setField((startFace,startFace+nFaces), 
                    faceFields[0][neighbourStartFace:neighbourStartFace+nFaces])
        faceFields[1] = faceExchange(faceFields[0], faceFields[1], self.procStartFace)

    def update(self, index, phi, gradPhi):
        pass

class FirstOrder(Reconstruct):
    def update(index, phi, gradPhi):
        C = self.faceOptions[index][0]
        phiC = phi.field[C]
        return 

class SecondOrder(Reconstruct):
    # weights selected in mesh.py
    def update(self, index, phi, gradPhi):
        mesh = phi.mesh
        C, D = self.faceOptions[index]
        phiC = phi.field[C]
        phiD = phi.field[D]
        phiDC = (phiD-phiC)*1

        gradF = Field('gradF({0})'.format(phi.name), phiDC, phi.dimensions)
        gradC = Field('gradC({0})'.format(phi.name), gradPhi.field[C], gradPhi.dimensions)

        weights1 = Field('lw', mesh.linearWeights[self.indices,index].reshape((-1,1)), (1,))
        weights2 = Field('qw', mesh.quadraticWeights[self.indices,:,index], (3,))
        return phiC + (weights1*gradF + gradC.dot(weights2)).field

class reduceAbsMinOp(T.Op):
    __props__ = ()
    def __init__(self):
        pass

    def make_node(self, x, y):
        assert hasattr(self, '_props')
        return T.Apply(self, [x, y], [x.type()])

    def perform(self, node, inputs, output_storage):
        field = inputs[0]
        count = inputs[1]
        output_storage[0][0] = compat.reduceAbsMin(field, count)
reduceAbsMin = reduceAbsMinOp()

class selectMultipleRangeOp(T.Op):
    __props__ = ()
    def __init__(self):
        pass

    def make_node(self, x, y):
        assert hasattr(self, '_props')
        return T.Apply(self, [x, y], [x.type()])

    def perform(self, node, inputs, output_storage):
        start = inputs[0]
        count = inputs[1]
        output_storage[0][0] = compat.selectMultipleRange(start, count)
selectMultipleRange = selectMultipleRangeOp()

class ENO(Reconstruct):
    def __init__(self, solver):
        super(ENO, self).__init__(solver)
        mesh = self.mesh
        meshO = mesh.origMesh
        combinations = np.array(list(itertools.combinations(range(0, 6), 3)))
        enoCount = np.zeros(meshO.volumes.shape[0], np.int32)
        enoIndices = []
        distDets = []
        neighbourDists = []
        for index, triplet in enumerate(combinations):
            neighbours =  meshO.cellNeighbours[:,triplet]
            neighbourDist = meshO.cellCentres[neighbours]-np.expand_dims(meshO.cellCentres[:meshO.nInternalCells], 1)
            distDet = np.linalg.det(neighbourDist)
            enoIndices.append((np.abs(distDet) > 1e-4*meshO.volumes.flatten()))
            enoCount += enoIndices[-1]
            distDets.append(distDet)
            neighbourDists.append(neighbourDist)
        enoIndices = np.vstack(enoIndices).T
        distDets = np.vstack(distDets).T
        neighbourDists = np.vstack(np.expand_dims(neighbourDists,axis=0)).transpose((1,0,2,3))

        indices = self.valueIndices[0]
        nIF = meshO.nInternalFaces
        self.enoCount = []
        self.enoIndices = []
        self.faceDistsDets = []
        self.distDets = []
        for faces, C in [(indices, meshO.owner[indices]), (indices[:nIF], meshO.neighbour[indices[:nIF]])]:
            enoFaceCount = enoCount[C]
            enoFaceIndices = np.nonzero(enoIndices[C])[1]
            faceDists = meshO.faceCentres[faces]-meshO.cellCentres[C]
            faceDists = np.repeat(faceDists, enoFaceCount, axis=0)
            faceIndices = np.repeat(C, enoFaceCount)

            faceNeighbourDists = neighbourDists[faceIndices, enoFaceIndices]
            faceDistsDets = []
            for index in range(0, 3):
                faceNeighbourDistsTemp = faceNeighbourDists.copy()
                faceNeighbourDistsTemp[:,index,:] = faceDists
                faceDistsDets.append(np.linalg.det(faceNeighbourDistsTemp).reshape(-1,1))

            self.solver.postpro.append((ad.ivector(), enoFaceCount.astype(np.int32)))
            self.enoCount.append(solver.postpro[-1][0])
            self.solver.postpro.append((ad.ivector(), enoFaceIndices.astype(np.int32)))
            self.enoIndices.append(solver.postpro[-1][0])
            #tmp =  np.abs(np.concatenate(faceDistsDets))
            self.solver.postpro.append((ad.bctensor3(), np.expand_dims(np.concatenate(faceDistsDets, axis=1), 2).astype(config.precision)))
            self.faceDistsDets.append(solver.postpro[-1][0])
            self.solver.postpro.append((ad.bcmatrix(), np.expand_dims(distDets[faceIndices, enoFaceIndices], 1).astype(config.precision)))
            self.distDets.append(solver.postpro[-1][0])

        self.solver.postpro.append((ad.imatrix(), combinations.astype(np.int32)))
        self.combinations = solver.postpro[-1][0]

        nIF = self.mesh.nInternalFaces
        for index in range(0, 2):
            self.faceOptions[1][index] = self.faceOptions[1][index][:nIF]
        self.Iindices = [self.indices, self.indices[:nIF]]
        self.BCUpdate = True
        
        self.enoStartCount = []
        for index in range(0, 2):
            enoCount = self.enoCount[index]
            enoStartCount = enoCount.cumsum()-enoCount
            self.enoStartCount.append(enoStartCount)
        return

    def partialUpdate(self, index, indices, phi, gradPhi):
        C = self.faceOptions[index][0][indices]
        enoCount = self.enoCount[index][indices]
        enoStartCount = self.enoStartCount[index][indices]
        faceIndices = selectMultipleRange(enoStartCount, enoCount)
        
        enoIndices = self.enoIndices[index][faceIndices]
        faceDistsDets = self.faceDistsDets[index][faceIndices]
        distDets = self.distDets[index][faceIndices]

        phiC = phi.field[C]
        combinations = self.combinations[enoIndices]
        repIndices = ad.repeat(C, enoCount).reshape((-1,1))
        phiN = phi.field[self.mesh.cellNeighbours[repIndices, combinations]]
        phiP = phi.field[repIndices]
        dphi = phiN-phiP
        dphi = (dphi*faceDistsDets).sum(axis=1)/distDets

        dphi = reduceAbsMin(dphi, enoCount)
        dphi = ad.patternbroadcast(dphi, phi.field.broadcastable)
        return phiC + dphi

    def update(self, index, phi, gradPhi):
        return self.partialUpdate(index, self.Iindices[index], phi, gradPhi)

class limitedSecondOrder(Reconstruct):
    def update(self, index, phi, gradPhi):
        R = Field('R', mesh.cellCentres[D] - mesh.cellCentres[C], (3,))
        F = Field('F', faceCentres - mesh.cellCentres[C], (3,))
        weights = (F.dot(R))/(R.dot(R))
        gradC = gradC.dot(R)
        if len(gradPhi.dimensions) == 2:
            gradC = gradC.dot(gradF)
            gradF = gradF.magSqr()
        r = 2.*gradC/gradF.stabilise(config.SMALL) - 1.
        #r = Field.switch(ad.gt(gradC.abs().field, 1000.*gradF.abs().field), 2.*1000.*gradC.sign()*gradF.sign() - 1., 2.*gradC/gradF.stabilise(config.VSMALL) - 1.)
        psi = lambda r, rabs: (r + rabs)/(1 + rabs)
        limiter = psi(r, r.abs()).field
        return phiC + weights.field*limiter*phiDC

class AnkitENO(SecondOrder):
    def __init__(self, solver):
        super(AnkitENO, self).__init__(solver)
        self.ENO = ENO(solver)
        mesh = self.mesh

        faceMap = -ad.ones_like(mesh.owner)
        faceMap = ad.set_subtensor(faceMap[:mesh.nInternalFaces], ad.arange(0, mesh.nInternalFaces))
        for patchID in mesh.localPatches:
            patch = mesh.boundary[patchID]
            if patch['type'] in config.cyclicPatches:
                startFace = patch['startFace']
                nFaces = patch['nFaces']
                faceMap = ad.set_subtensor(faceMap[startFace:startFace+nFaces], \
                    ad.arange(self.ENO.cyclicStartFaces[patchID],self.ENO.cyclicStartFaces[patchID]+nFaces))
        nLocalFaces = mesh.nLocalCells - mesh.nInternalCells + mesh.nInternalFaces
        nRemoteFaces = mesh.nFaces-nLocalFaces
        faceMap = ad.set_subtensor(faceMap[nLocalFaces:], \
                ad.arange(self.ENO.procStartFace, self.ENO.procStartFace + nRemoteFaces))
        self.faceMap = faceMap
        return

    def shockSensor(self, (UFs, TFs, pFs), (U, T, p), (gradU, _, __)):
        shock = ad.zeros_like(self.indices)
        for index, (C, D) in enumerate(self.ENO.faceOptions):
            gradULF = gradU.getField(C)
            dil = gradULF.trace().field
            dil2 = dil*dil
            gradULF = gradULF.field
            wx = gradULF[:,2,1]-gradULF[:,1,2]
            wy = gradULF[:,0,2]-gradULF[:,2,0]
            wz = gradULF[:,1,0]-gradULF[:,0,1]
            w2 = wx*wx + wy*wy + wz*wz
            w2 = w2.reshape((-1,1))
            delta = self.mesh.volumes[C]**(1./3)
            sos = ad.sqrt(self.solver.gamma*self.solver.R*T.field[C])
            C_SS = 0.5*(1.-ad.tanh(2.5+10.*delta/sos*dil))*(dil2/(dil2+w2+1e-7))
            pF = pFs[index].field[self.faceMap[self.ENO.indices[index]]]
            condition = ad.abs_(pF-p.field[C]) > 0.99*ad.minimum(p.field[C],p.field[D])
            shock = ad.set_subtensor(shock[condition.nonzero()[0]], 1)
            #shock[ad.abs_] = 1
            condition = C_SS > 0.02
            shock = ad.set_subtensor(shock[condition.nonzero()[0]], 1)
        # expand by 1 face
        for index in range(0, 1):
            shockIndices = shock.nonzero()[0]
            owner, neighbour = self.faceOptions[0]
            cells = neighbour[shockIndices]
            condition = cells < self.mesh.nInternalCells
            cells = cells[condition.nonzero()[0]]
            cells = ad.concatenate((cells, owner[shockIndices]))
            faces = self.mesh.cellFaces[cells].flatten()
            condition = faces > -1
            faces = self.faceMap[faces[condition.nonzero()[0]]]
            shock = ad.set_subtensor(shock[faces], 1)
        self.solver.local = (ad.extra_ops.Unique(cells).shape[0]*1.)/self.mesh.nInternalCells
        #local = ad.zeros_like(self.mesh.volumes)
        #local = ad.set_subtensor(local[cells], 1)
        #self.solver.local = local
        return shock

    def dualSystem(self, *fieldsSys):
        faceFieldsSys = super(AnkitENO, self).dualSystem(*fieldsSys)
        (U, gradU), (T, gradT), (p, gradp) = fieldsSys
        shock = self.shockSensor(faceFieldsSys, (U, T, p), (gradU, gradT, gradp))
        shockIndices = shock.nonzero()[0]
        shockIndices = [shockIndices, shockIndices.copy()]
        condition = shockIndices[1] < self.mesh.nInternalFaces
        shockIndices[1] = shockIndices[1][condition.nonzero()[0]]

        for phiFs, (phi, gradPhi) in zip(faceFieldsSys, fieldsSys):
            for index in range(0, 2):
                phiFs[index].setField(shockIndices[index],
                    self.ENO.partialUpdate(index, shockIndices[index], phi, gradPhi) 
                )
            # overwrites work done by second order
            self.faceBCUpdate(phiFs)

        return faceFieldsSys

#def TVD_dual(phi, gradPhi):
#    from op import grad
#    assert len(phi.dimensions) == 1
#    logger.info('TVD {0}'.format(phi.name))
#    mesh = phi.mesh
#
#    # every face gets filled
#    faceField = ad.bcalloc(config.precision(0.), (mesh.nFaces, phi.dimensions[0]))
#    faceFields = [faceField, faceField.copy()]
#    # van leer
#    psi = lambda r, rabs: (r + rabs)/(1 + rabs)
#    def update(start, end, index, copy=True):
#        owner = mesh.owner[start:end]
#        neighbour = mesh.neighbour[start:end]
#        faceCentres = mesh.faceCentres[start:end]
#        deltas = mesh.deltas[start:end]
#        if index == 0:
#            C, D = [owner, neighbour]
#        else:
#            C, D = [neighbour, owner]
#        phiC = phi.field[C]
#        phiD = phi.field[D]
#        # wTF is *1 necessary over here for theano
#        phiDC = (phiD-phiC)*1
#        R = Field('R', mesh.cellCentres[D] - mesh.cellCentres[C], (3,))
#        F = Field('F', faceCentres - mesh.cellCentres[C], (3,))
#        gradC = Field('gradC({0})'.format(phi.name), gradPhi.field[C], gradPhi.dimensions)
#        gradF = Field('gradF({0})'.format(phi.name), phiDC, phi.dimensions)
#        gradC = gradC.dot(R)
#        if phi.dimensions[0] == 3:
#            gradC = gradC.dot(gradF)
#            gradF = gradF.magSqr()
#        r = 2.*gradC/gradF.stabilise(config.SMALL) - 1.
#        #r = Field.switch(ad.gt(gradC.abs().field, 1000.*gradF.abs().field), 2.*1000.*gradC.sign()*gradF.sign() - 1., 2.*gradC/gradF.stabilise(config.VSMALL) - 1.)
#
#        weights = (F.dot(R)).field/(deltas*deltas)
#        #weights = 0.5
#        limiter = psi(r, r.abs()).field
#        #phi.solver.local = phiC
#        #phi.solver.remote = phiD
#        faceFields[index] = ad.set_subtensor(faceFields[index][start:end], phiC + weights*limiter*phiDC)
#    # internal, then local patches and finally remote
#    update(0, mesh.nInternalFaces, 0)
#    update(0, mesh.nInternalFaces, 1)
#    for patchID in mesh.localPatches:
#        startFace = mesh.boundary[patchID]['startFace']
#        endFace = startFace + mesh.boundary[patchID]['nFaces']
#        patchType = mesh.boundary[patchID]['type']
#        if patchType in config.cyclicPatches:
#            update(startFace, endFace, 0)
#            update(startFace, endFace, 1)
#        elif patchType == 'characteristic':
#            #update(startFace, endFace, 0)
#            owner = mesh.owner[startFace:endFace]
#            phiC = phi.field[owner]
#            R = mesh.faceCentres[startFace:endFace]-mesh.cellCentres[owner]
#            F = Field('F', mesh.faceCentres[startFace:endFace] - mesh.cellCentres[owner], (3,))
#            gradC = Field('gradC({0})'.format(phi.name), gradPhi.field[owner], gradPhi.dimensions)
#            faceFields[0] = ad.set_subtensor(faceFields[0][startFace:endFace], phiC + (gradPhi.dot(F)).field)
#            faceFields[1] = ad.set_subtensor(faceFields[1][startFace:endFace], phi.field[mesh.neighbour[startFace:endFace]])
#        else:
#            for index in range(0, 2):
#                faceFields[index] = ad.set_subtensor(faceFields[index][startFace:endFace], phi.field[mesh.neighbour[startFace:endFace]])
#    nRemoteFaces = mesh.nFaces-(mesh.nCells-mesh.nLocalCells)
#    update(nRemoteFaces, mesh.nFaces, 0, False)
#    update(nRemoteFaces, mesh.nFaces, 1, False)
#
#    #if phi.name == 'T':
#    #    phi.solver.local = phi.field
#    #    phi.solver.remote = gradPhi.field
#
#    return [Field('{0}F'.format(phi.name), faceField, phi.dimensions) for faceField in faceFields]

# will not work
#def upwind(phi, U): 
#    assert len(phi.dimensions) == 1
#    logger.info('upwinding {0} using {1}'.format(phi.name, U.name)) 
#    mesh = phi.mesh
#    faceField = ad.bcalloc(config.precision(0.), (mesh.nFaces, phi.dimensions[0]))
#    def update(start, end):
#        positiveFlux = ad.sum(U.field[start:end] * mesh.normals[start:end], axis=1) > 0
#        negativeFlux = 1 - positiveFlux
#        faceField[positiveFlux] = phi.field[mesh.owner[positiveFlux]]
#        faceField[negativeFlux] = phi.field[mesh.neighbour[negativeFlux]]
#
#    update(0, mesh.nInternalFaces)
#    for patchID in mesh.localPatches:
#        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
#        if phi.boundary[patchID]['type'] in config.cyclicPatches:
#            update(startFace, endFace)
#        else:
#            faceField[startFace:endFace] = phi.field[mesh.neighbour[startFace:endFace]]
#    update(mesh.nFaces-(mesh.nCells-mesh.nLocalCells), mesh.nFaces)
#
#    return Field('{0}F'.format(phi.name), faceField, phi.dimensions)
#

