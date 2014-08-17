from __future__ import print_function
import numpy as np
import numpad as ad
import time

from field import Field, FaceField
import utils
logger = utils.logger(__name__)

def TVD(field, U):
    logger.info('TVD {0} using {1}'.format(field.name, U.name))
    mesh = field.mesh
    positiveFlux = ad.value(ad.sum(U.field * mesh.normals, axis=1)) > 0
    negativeFlux = (positiveFlux == False)

    # van leer
    psi = lambda r: ad.adarray((r + np.abs(r))/(1+np.abs(r)))

    faceField = ad.zeros((mesh.nFaces, field.field.shape[1]))
    # restructure, cyclic boundaries?
    for C, D, flux in [[mesh.owner[positiveFlux], mesh.neighbour[positiveFlux], positiveFlux], [mesh.neighbour[negativeFlux], mesh.owner[negativeFlux], negativeFlux]]:
        def gradient(x):
            gradx = grad(interpolate(x))
            return FaceField('grad' + x.name, mesh, ad.concatenate((gradx, ad.zeros((mesh.nGhostCells, 3))))[C])
        R = FaceField('R', mesh, mesh.cellCentres[D] - mesh.cellCentres[C])
        gradF = FaceField('gradF', mesh, field.field[D]-field.field[C])
        if field.field.shape[1] == 1:
            gradC = gradient(field)
            r = 2*gradC.dot(R)/(gradF + 1e-16) - 1
        else:
            gradCxdotR = gradient(field.component(0)).dot(R)
            gradCydotR = gradient(field.component(1)).dot(R)
            gradCzdotR = gradient(field.component(2)).dot(R)
            gradCdotR = FaceField('gradCdotR', mesh, ad.hstack((gradCxdotR.field, gradCydotR.field, gradCzdotR.field)))
            r = 2*gradCdotR.dot(gradF)/(gradF.magSqr() + 1e-16) - 1
        #if r.field.shape[0] > 0:
        #    r.info()
        r = ad.value(r.field)
        faceField[flux] = field.field[C] + 0.5*psi(r)*(field.field[D] - field.field[C])

    # correction for ghost cells
    for patchID in field.boundary:
        if field.boundary[patchID]['type'] != 'cyclic':
            startFace = mesh.boundary[patchID]['startFace']
            endFace = startFace + mesh.boundary[patchID]['nFaces']
            faceField[startFace:endFace] = field.field[mesh.neighbour[startFace:endFace]]

    return FaceField(field.name + 'F', mesh, faceField)

def upwind(field, U):
    logger.info('upwinding {0} using {1}'.format(field.name, U.name))
    mesh = field.mesh
    positiveFlux = ad.value(ad.sum(U.field * mesh.normals, axis=1)) > 0
    negativeFlux = (positiveFlux == False)
    faceField = ad.zeros((mesh.nFaces, field.field.shape[1]))
    faceField[positiveFlux] = field.field[mesh.owner[positiveFlux]]
    faceField[negativeFlux] = field.field[mesh.neighbour[negativeFlux]]

    for patchID in field.boundary:
        if field.boundary[patchID]['type'] != 'cyclic':
            startFace = mesh.boundary[patchID]['startFace']
            endFace = startFace + mesh.boundary[patchID]['nFaces']
            faceField[startFace:endFace] = field.field[mesh.neighbour[startFace:endFace]]

    return FaceField(field.name + 'F', mesh, faceField)

def interpolate(field):
    logger.info('interpolating {0}'.format(field.name))
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas)
    faceField = FaceField(field.name + 'F', mesh, field.field[mesh.owner]*factor + field.field[mesh.neighbour]*(1-factor))
    return faceField

def div(field, U=None):
    logger.info('divergence of {0}'.format(field.name))
    mesh = field.mesh
    if U == None:
        return (mesh.sumOp * (field.field * mesh.areas))/mesh.volumes
    else:
        return (mesh.sumOp * ((field * U).dotN().field * mesh.areas))/mesh.volumes

def grad(field):
    logger.info('gradient of {0}'.format(field.name))
    mesh = field.mesh
    return (mesh.sumOp * (field.field * mesh.normals * mesh.areas))/mesh.volumes

def laplacian(field, DT):
    logger.info('laplacian of {0}'.format(field.name))
    mesh = field.mesh

    # non orthogonal correction
    #DTgradF = Field.zeros('grad' + field.name, mesh, mesh.nCells, 3.)
    #DTgradF.setInternalField(DT*grad(field))
    #laplacian1 = div(interpolate(DTgradF), 1.)

    gradFdotn = (field.field[mesh.neighbour]-field.field[mesh.owner])/mesh.deltas
    laplacian2 = (mesh.sumOp * (DT * gradFdotn * mesh.areas))/mesh.volumes
    return laplacian2

def ddt(field, field0, dt):
    logger.info('ddt of {0}'.format(field.name))
    return (field.getInternalField()-ad.value(field0.getInternalField()))/dt

def explicit(equation, boundary, fields, dt):
    names = [field.name for field in fields]
    print('Time marching for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].info()
     
    start = time.time()
    
    LHS = equation(*fields)
    internalFields = [(fields[index].getInternalField() - LHS[index]*dt) for index in range(0, len(fields))]
    boundary(*internalFields)

    end = time.time()
    print('Time for iteration:', end-start)
 

def implicit(equation, boundary, fields):
    names = [field.name for field in fields]
    print('Solving for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].info()

    start = time.time()

    nDims = [field.field.shape[1] for field in fields]
    def setInternalFields(stackedInternalFields):
        curr = 0
        internalFields = []
        for index in range(0, len(fields)):
            internalFields.append(stackedInternalFields[:,curr:curr+nDims[index]])
            curr += nDims[index]
        boundary(*internalFields)

    def solver(internalFields):
        setInternalFields(internalFields)
        return ad.hstack(equation(*fields))

    stack = [field.getInternalField() for field in fields]
    solution = ad.solve(solver, ad.hstack(stack))
    setInternalFields(solution)

    end = time.time()
    print('Time for iteration:', end-start)

def forget(fields):
    logger.info('forgetting fields')
    for field in fields:
        field.field.obliviate()

