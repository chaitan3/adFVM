from __future__ import print_function
import numpy as np
import numpad as ad
import time

from field import Field, FaceField
import utils
logger = utils.logger(__name__)

def interpolate(field):
    logger.info('interpolating {0}'.format(field.name))
    mesh = field.mesh
    factor = (mesh.faceDeltas/mesh.deltas)
    faceField = FaceField(field.name + 'f', mesh, field.field[mesh.owner]*factor + field.field[mesh.neighbour]*(1-factor))
    return faceField

def div(field, U):
    logger.info('divergence of {0}'.format(field.name))
    mesh = field.mesh
    Fdotn = ad.sum((field.field[:,np.newaxis,:] * U.field[:,:,np.newaxis]) * mesh.normals[:,:,np.newaxis], axis=1)
    return (mesh.sumOp * (Fdotn * mesh.areas))/mesh.volumes

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

def solve(equation, fields):
    names = [field.name for field in fields]
    print('Solving for', ' '.join(names))

    start = time.time()

    nDims = [field.field.shape[1] for field in fields]
    def setInternalFields(internalFields):
        curr = 0
        for index in range(0, len(fields)):
            fields[index].setInternalField(internalFields[:,curr:curr+nDims[index]])

    def solver(internalFields):
        setInternalFields(internalFields)
        return ad.hstack(equation(*fields))

    stack = [field.getInternalField() for field in fields]
    solution = ad.solve(solver, ad.hstack(stack))
    setInternalFields(solution)

    end = time.time()
    print('Time for iteration:', end-start)

