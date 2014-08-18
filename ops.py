from __future__ import print_function
import numpy as np
import numpad as ad
import time

from field import Field
import utils
logger = utils.logger(__name__)

def TVD_dual(field):
    logger.info('TVD {0}'.format(field.name))
    mesh = field.mesh
    # van leer
    psi = lambda r: (r + r.abs())/(1 + r.abs())
    SMALL = 1e-16

    faceFields = []
    # restructure, cyclic boundaries?
    owner = mesh.owner[:mesh.nInternalFaces]
    neighbour = mesh.neighbour[:mesh.nInternalFaces]
    for C, D in [[owner, neighbour], [neighbour, owner]]:
        def gradient(x):
            gradx = grad(interpolate(x))
            return Field('grad' + x.name, mesh, gradx.field[C])
        R = Field('R', mesh, mesh.cellCentres[D] - mesh.cellCentres[C])
        gradF = Field('gradF', mesh, field.field[D]-field.field[C])
        if field.field.shape[1] == 1:
            gradC = gradient(field)
            r = 2*gradC.dot(R)/(gradF + SMALL) - 1
        else:
            gradCxdotR = gradient(field.component(0)).dot(R)
            gradCydotR = gradient(field.component(1)).dot(R)
            gradCzdotR = gradient(field.component(2)).dot(R)
            gradCdotR = Field('gradCdotR', mesh, ad.hstack((gradCxdotR.field, gradCydotR.field, gradCzdotR.field)))
            r = 2*gradCdotR.dot(gradF)/(gradF.magSqr() + SMALL) - 1
        faceFields.append(Field(field.name + 'F', mesh, ad.zeros((mesh.nFaces, field.field.shape[1]))))
        faceFields[-1].field[:mesh.nInternalFaces] = field.field[C] + 0.5*psi(r).field*(field.field[D] - field.field[C])

        # correction for ghost cells
        for patchID in field.boundary:
            if field.boundary[patchID]['type'] != 'cyclic':
                startFace = mesh.boundary[patchID]['startFace']
                endFace = startFace + mesh.boundary[patchID]['nFaces']
                faceFields[-1].field[startFace:endFace] = field.field[mesh.neighbour[startFace:endFace]]

    return faceFields

def upwind(field, U):
    logger.info('upwinding {0} using {1}'.format(field.name, U.name))
    mesh = field.mesh
    faceField = ad.zeros((mesh.nFaces, field.field.shape[1]))
    def update(start, end):
        positiveFlux = ad.value(ad.sum(U.field * mesh.normals, axis=1)) > 0
        negativeFlux = (positiveFlux == False)
        faceField[positiveFlux] = field.field[mesh.owner[positiveFlux]]
        faceField[negativeFlux] = field.field[mesh.neighbour[negativeFlux]]

    update(0, mesh.nInternalFaces)
    for patchID in field.boundary:
        startFace = mesh.boundary[patchID]['startFace']
        endFace = startFace + mesh.boundary[patchID]['nFaces']
        if field.boundary[patchID]['type'] != 'cyclic':
            faceField[startFace:endFace] = field.field[mesh.neighbour[startFace:endFace]]
        else:
            update(startFace, endFace)

    return Field(field.name + 'F', mesh, faceField)

def interpolate(field):
    logger.info('interpolating {0}'.format(field.name))
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas)
    faceField = Field(field.name + 'F', mesh, field.field[mesh.owner]*factor + field.field[mesh.neighbour]*(1-factor))
    return faceField

def div(field, U=None):
    logger.info('divergence of {0}'.format(field.name))
    mesh = field.mesh
    if U == None:
        divField = (mesh.sumOp * (field.field * mesh.areas))/mesh.volumes
    else:
        divField = (mesh.sumOp * ((field * U).dotN().field * mesh.areas))/mesh.volumes
    return Field('div' + field.name, mesh, divField)

def grad(field):
    logger.info('gradient of {0}'.format(field.name))
    mesh = field.mesh
    gradField = (mesh.sumOp * (field.field * mesh.normals * mesh.areas))/mesh.volumes
    return Field('grad' + field.name, mesh, gradField)

def laplacian(field, DT):
    logger.info('laplacian of {0}'.format(field.name))
    mesh = field.mesh

    # non orthogonal correction
    #DTgradF = Field.zeros('grad' + field.name, mesh, mesh.nCells, 3.)
    #DTgradF.setInternalField(DT*grad(field))
    #laplacian1 = div(interpolate(DTgradF), 1.)

    gradFdotn = (field.field[mesh.neighbour]-field.field[mesh.owner])/mesh.deltas
    laplacian2 = (mesh.sumOp * (DT * gradFdotn * mesh.areas))/mesh.volumes
    return Field('laplacian' + field.name, mesh, laplacian2)

def ddt(field, field0, dt):
    logger.info('ddt of {0}'.format(field.name))
    return Field('ddt' + field.name, field.mesh, (field.getInternalField()-ad.value(field0.getInternalField()))/dt)

def explicit(equation, boundary, fields, dt):
    names = [field.name for field in fields]
    print('Time marching for', ' '.join(names))
    for index in range(0, len(fields)):
        fields[index].info()
     
    start = time.time()
    
    LHS = equation(*fields)
    internalFields = [(fields[index].getInternalField() - LHS[index].field*dt) for index in range(0, len(fields))]
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
        equationFields = [field.field for field in equation(*fields)]
        return ad.hstack(equationFields)

    stack = [field.getInternalField() for field in fields]
    solution = ad.solve(solver, ad.hstack(stack))
    setInternalFields(solution)

    end = time.time()
    print('Time for iteration:', end-start)

def forget(fields):
    logger.info('forgetting fields')
    for field in fields:
        field.field.obliviate()

