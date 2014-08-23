from __future__ import print_function
import numpy as np
import numpad as ad
import time
import scipy.sparse as sp

from field import Field, CellField
import utils
logger = utils.logger(__name__)

def TVD_dual(phi):
    assert len(phi.dimensions) == 1
    logger.info('TVD {0}'.format(phi.name))
    mesh = phi.mesh
    # van leer
    psi = lambda r, rabs: (r + rabs)/(1 + rabs)

    faceField = ad.zeros((mesh.nFaces, phi.dimensions[0]))
    faceFields = [faceField, faceField.copy()]
    gradField = grad(phi, ghost=True)

    def update(start, end):
        owner = mesh.owner[start:end]
        neighbour = mesh.neighbour[start:end]
        index = 0
        for C, D in [[owner, neighbour], [neighbour, owner]]:
            phiC = phi.field[C]
            phiD = phi.field[D]
            phiDC = phiD-phiC
            R = Field('R', mesh, ad.array(mesh.cellCentres[D] - mesh.cellCentres[C]))
            gradC = Field('gradC({0})'.format(phi.name), mesh, gradField.field[C])
            gradF = Field('gradF({0})'.format(phi.name), mesh, phiDC)
            if phi.dimensions[0] == 1:
                r = 2*gradC.dot(R)/(gradF + utils.SMALL) - 1
            else:
                r = 2*gradC.dot(R).dot(gradF)/(gradF.magSqr() + utils.SMALL) - 1
            faceFields[index][start:end] = phiC + 0.5*psi(r, r.abs()).field*phiDC
            index += 1

    update(0, mesh.nInternalFaces)
    for patchID in phi.boundary:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        if phi.boundary[patchID]['type'] == 'cyclic':
            update(startFace, endFace)
        else:
            for faceField in faceFields:
                faceField[startFace:endFace] = phi.field[mesh.neighbour[startFace:endFace]]

    return [Field('{0}F'.format(phi.name), mesh, faceField) for faceField in faceFields]

def upwind(phi, U): 
    assert len(phi.dimensions) == 1
    logger.info('upwinding {0} using {1}'.format(phi.name, U.name)) 
    mesh = phi.mesh
    faceField = ad.zeros((mesh.nFaces, phi.dimensions[0]))
    def update(start, end):
        positiveFlux = ad.value(ad.sum(U.field[start:end] * mesh.normals[start:end], axis=1)) > 0
        negativeFlux = 1 - positiveFlux
        faceField[positiveFlux] = phi.field[mesh.owner[positiveFlux]]
        faceField[negativeFlux] = phi.field[mesh.neighbour[negativeFlux]]

    update(0, mesh.nInternalFaces)
    for patchID in phi.boundary:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        if phi.boundary[patchID]['type'] == 'cyclic':
            update(startFace, endFace)
        else:
            faceField[startFace:endFace] = phi.field[mesh.neighbour[startFace:endFace]]

    return Field('{0}F'.format(phi.name), mesh, faceField)

def interpolate(phi):
    logger.info('interpolating {0}'.format(phi.name))
    mesh = phi.mesh
    factor = mesh.weights
    # for tensor
    if len(factor.shape)-1 < len(phi.dimensions):
        factor = factor.reshape((factor.shape[0], 1, 1))
    faceField = Field('{0}F'.format(phi.name), mesh, phi.field[mesh.owner]*factor + phi.field[mesh.neighbour]*(1-factor))
    return faceField

def div(phi, U=None, ghost=False):
    logger.info('divergence of {0}'.format(phi.name))
    mesh = phi.mesh
    if phi.size == mesh.nCells:
        phi = interpolate(phi)
    if U is None:
        divField = (mesh.sumOp * (phi.field * mesh.areas))/mesh.volumes
    else:
        assert phi.dimensions == (1,)
        if U.size == mesh.nCells:
            U = interpolate(U)
        divField = (mesh.sumOp * ((phi * U).dotN().field * mesh.areas))/mesh.volumes
    if ghost:
        return CellField('div({0})'.format(phi.name), mesh, divField)
    else:
        return Field('div({0})'.format(phi.name), mesh, divField)

def grad(phi, ghost=False):
    assert len(phi.dimensions) == 1
    logger.info('gradient of {0}'.format(phi.name))
    mesh = phi.mesh
    if phi.size == mesh.nCells:
        phi = interpolate(phi)
    product = phi.outer(mesh.Normals)
    gradField = (mesh.sumOp * (product.field * mesh.areas[:,:,np.newaxis]))/mesh.volumes[:,:,np.newaxis]
    # if grad of scalar
    if phi.dimensions[0] == 1:
        gradField = gradField.reshape((mesh.nInternalCells, 3))
    if ghost:
        return CellField('grad({0})'.format(phi.name), mesh, gradField)
    else:
        return Field('grad({0})'.format(phi.name), mesh, gradField)

def snGrad(phi):
    logger.info('snGrad of {0}'.format(phi.name))
    mesh = phi.mesh
    gradFdotn = (phi.field[mesh.neighbour]-phi.field[mesh.owner])/mesh.deltas
    return Field('snGrad({0})'.format(phi.name), mesh, gradFdotn)

def laplacian(phi, DT):
    logger.info('laplacian of {0}'.format(phi.name))
    mesh = phi.mesh

    # non orthogonal correction
    #DTgradF = Field.zeros('grad' + phi.name, mesh, mesh.nCells, 3.)
    #DTgradF.setInternalField(DT*grad(field))
    #laplacian1 = div(interpolate(DTgradF), 1.)

    gradFdotn = snGrad(phi).field
    laplacian2 = (mesh.sumOp * (DT * gradFdotn * mesh.areas))/mesh.volumes
    return Field('laplacian({0})'.format(phi.name), mesh, laplacian2)

def ddt(phi, phi0, dt):
    logger.info('ddt of {0}'.format(phi.name))
    return Field('ddt' + phi.name, phi.mesh, (phi.getInternalField()-ad.value(phi0.getInternalField()))/dt)

def explicit(equation, boundary, fields, solver):
    start = time.time()

    names = [phi.name for phi in fields]
    print('Time marching for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = CellField.copy(fields[index])
        fields[index].info()
    
    LHS = equation(*fields)
    internalFields = [(fields[index].getInternalField() - LHS[index].field*solver.dt) for index in range(0, len(fields))]
    newFields = boundary(*internalFields)
    for index in range(0, len(fields)):
        newFields[index].name = fields[index].name
        #newFields[index].old = fields[index]

    end = time.time()
    print('Time for iteration:', end-start)
    return newFields

def implicit(equation, boundary, fields, dt):
    start = time.time()

    names = [phi.name for phi in fields]
    print('Solving for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].old = CellField.copy(fields[index])
        fields[index].info()

    nDimensions = np.concatenate(([0], np.cumsum(np.array([phi.dimensions[0] for phi in adjointFields]))))
    nDimensions = zip(nDimensions[:-1], nDimensions[1:])

    def setInternalFields(stackedInternalFields):
        location = 0
        internalFields = []
        for index in range(0, len(fields)):
            internalFields.append(stackedInternalFields[:, range(*nDimensions[index])])
        boundary(*internalFields)

    def solver(internalFields):
        setInternalFields(internalFields)
        equationFields = [phi.field for phi in equation(*fields)]
        return ad.hstack(equationFields)

    stack = [phi.getInternalField() for phi in fields]
    solution = ad.solve(solver, ad.hstack(stack))
    setInternalFields(solution)

    end = time.time()
    print('Time for iteration:', end-start)

def derivative(newField, oldFields):
    start = time.time()

    names = [phi.name for phi in oldFields]
    diffs = []
    for phi in oldFields:
        diffs.append(newField.diff(phi.field))
    result = sp.hstack(diffs).toarray().ravel()

    end = time.time()
    print('Time for computing derivative:', end-start)
    return result

def forget(fields):
    logger.info('forgetting fields')
    for phi in fields:
        phi.field.obliviate()

def strip(fields):
    logger.info('forgetting fields')
    newFields = [CellField.copy(phi) for phi in fields]
    for phi in newFields:
        phi.field = ad.array(ad.value(phi.field))
    return newFields

