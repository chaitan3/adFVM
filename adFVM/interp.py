from . import config
from .config import ad
from field import Field

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
    def __init__(self, mesh, update, limiter=True):
        self.update = update
        self.mesh = mesh
        indices = [ad.arange(0, mesh.nInternalFaces)]
        Cindices = []
        Bindices = []
        for patchID in mesh.localPatches:
            startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
            patchType = mesh.boundary[patchID]['type']
            if patchType in config.cyclicPatches:
                indices.append(ad.arange(startFace, endFace))
            elif patchType == 'characteristic':
                Cindices.append(ad.arange(startFace, endFace))
            else:
                Bindices.append(ad.arange(startFace, endFace))
        nRemoteFaces = mesh.nFaces-(mesh.nCells-mesh.nLocalCells)
        indices.append(ad.arange(nRemoteFaces, mesh.nFaces))
        self.indices = ad.concatenate(indices)
        self.Cindices = Cindices
        self.Bindices = Bindices
        self.limiter = limiter

    def dual(self, phi, gradPhi):
        assert len(phi.dimensions) == 1
        logger.info('TVD {0}'.format(phi.name))

        faceFields = []
        faceFields.append(self.update(self.indices, 0, phi, gradPhi, self.limiter))
        faceFields.append(self.update(self.indices, 1, phi, gradPhi, self.limiter))

        return [Field('{0}F'.format(phi.name), faceField, phi.dimensions) for faceField in faceFields]

    # every face gets filled
    #faceField = ad.bcalloc(config.precision(0.), (mesh.nFaces, phi.dimensions[0]))
    #faceFields = [faceField, faceField.copy()]
    #update(0, mesh.nInternalFaces, 0, faceFields, phi, gradPhi)
    #update(0, mesh.nInternalFaces, 1, faceFields, phi, gradPhi)
    #for patchID in mesh.localPatches:
    #    startFace = mesh.boundary[patchID]['startFace']
    #    endFace = startFace + mesh.boundary[patchID]['nFaces']
    #    patchType = mesh.boundary[patchID]['type']
    #    if patchType in config.cyclicPatches:
    #        update(startFace, endFace, 0, faceFields, phi, gradPhi)
    #        update(startFace, endFace, 1, faceFields, phi, gradPhi)
    #    elif patchType == 'characteristic':
    #        # TVD
    #        #update(startFace, endFace, 0)
    #        # charles
    #        #faceFields[0] = characteristic(startFace, endFace, faceFields[0], phi, gradPhi)
    #        # first order
    #        faceFields[0] = ad.set_subtensor(faceFields[0][startFace:endFace], phi.field[mesh.owner[startFace:endFace]])
    #        faceFields[1] = ad.set_subtensor(faceFields[1][startFace:endFace], phi.field[mesh.neighbour[startFace:endFace]])
    #    else:
    #        for index in range(0, 2):
    #            faceFields[index] = ad.set_subtensor(faceFields[index][startFace:endFace], phi.field[mesh.neighbour[startFace:endFace]])
    #nRemoteFaces = mesh.nFaces-(mesh.nCells-mesh.nLocalCells)
    #update(nRemoteFaces, mesh.nFaces, 0, faceFields, phi, gradPhi)
    #update(nRemoteFaces, mesh.nFaces, 1, faceFields, phi, gradPhi)

    #if phi.name == 'T':
    #    phi.solver.local = phi.field
    #    phi.solver.remote = gradPhi.field

def characteristic(startFace, endFace, faceField, phi, gradPhi):
    mesh = phi.mesh
    owner = mesh.owner[startFace:endFace]
    phiC = phi.field[owner]
    F = Field('F', mesh.faceCentres[startFace:endFace] - mesh.cellCentres[owner], (3,))
    gradC = Field('gradC({0})'.format(phi.name), gradPhi.field[owner], gradPhi.dimensions)
    return ad.set_subtensor(faceField[startFace:endFace], phiC + (gradC.dot(F)).field)

#def TVD(start, end, index, faceFields, phi, gradPhi):
#    mesh = phi.mesh
#    owner = mesh.owner[start:end]
#    neighbour = mesh.neighbour[start:end]
#    faceCentres = mesh.faceCentres[start:end]
#    deltas = mesh.deltas[start:end]
def TVD(indices, index, phi, gradPhi, limiter):
    mesh = phi.mesh
    owner = mesh.owner[indices]
    neighbour = mesh.neighbour[indices]
    faceCentres = mesh.faceCentres[indices]
    if index == 0:
        C, D = [owner, neighbour]
    else:
        C, D = [neighbour, owner]
    phiC = phi.field[C]
    phiD = phi.field[D]
    phiDC = (phiD-phiC)*1

    R = Field('R', mesh.cellCentres[D] - mesh.cellCentres[C], (3,))
    F = Field('F', faceCentres - mesh.cellCentres[C], (3,))
    weights = (F.dot(R))/(R.dot(R))
    #weights = 0.5
    gradF = Field('gradF({0})'.format(phi.name), phiDC, phi.dimensions)
    gradC = Field('gradC({0})'.format(phi.name), gradPhi.field[C], gradPhi.dimensions)

    if limiter:
        gradC = gradC.dot(R)
        if len(gradPhi.dimensions) == 2:
            gradC = gradC.dot(gradF)
            gradF = gradF.magSqr()
        r = 2.*gradC/gradF.stabilise(config.SMALL) - 1.
        #r = Field.switch(ad.gt(gradC.abs().field, 1000.*gradF.abs().field), 2.*1000.*gradC.sign()*gradF.sign() - 1., 2.*gradC/gradF.stabilise(config.VSMALL) - 1.)
        psi = lambda r, rabs: (r + rabs)/(1 + rabs)
        limiter = psi(r, r.abs()).field
        return phiC + weights*limiter*phiDC
    else:
        # blending
        weights1 = weights
        weights2 = 0.*F
        #weights1 = 1./3*weights
        #weights2 = 2./3*F + 1./3*weights*R
        #weights1 = 0.
        #weights2 = F
        return phiC + (weights1*gradF + gradC.dot(weights2)).field


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
def upwind(phi, U): 
    assert len(phi.dimensions) == 1
    logger.info('upwinding {0} using {1}'.format(phi.name, U.name)) 
    mesh = phi.mesh
    faceField = ad.bcalloc(config.precision(0.), (mesh.nFaces, phi.dimensions[0]))
    def update(start, end):
        positiveFlux = ad.sum(U.field[start:end] * mesh.normals[start:end], axis=1) > 0
        negativeFlux = 1 - positiveFlux
        faceField[positiveFlux] = phi.field[mesh.owner[positiveFlux]]
        faceField[negativeFlux] = phi.field[mesh.neighbour[negativeFlux]]

    update(0, mesh.nInternalFaces)
    for patchID in mesh.localPatches:
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
        if phi.boundary[patchID]['type'] in config.cyclicPatches:
            update(startFace, endFace)
        else:
            faceField[startFace:endFace] = phi.field[mesh.neighbour[startFace:endFace]]
    update(mesh.nFaces-(mesh.nCells-mesh.nLocalCells), mesh.nFaces)

    return Field('{0}F'.format(phi.name), faceField, phi.dimensions)


