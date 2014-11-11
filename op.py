from __future__ import print_function
import numpy as np
import time

from field import Field, CellField
from interp import interpolate
from config import ad, Logger, adsparse
logger = Logger(__name__)

def internal_sum(phi):
    mesh = phi.mesh
    return (adsparse.basic.dot(mesh.sumOp, (phi.field * mesh.areas)))/mesh.volumes

def div(phi, U=None, ghost=False):
    logger.info('divergence of {0}'.format(phi.name))
    mesh = phi.mesh
    if U is None:
        divField = internal_sum(phi)
    else:
        assert phi.dimensions == (1,)
        raise Exception('not tested')
        divField = internal_sum((phi*U).dotN())
    if ghost:
        return CellField('div({0})'.format(phi.name), divField, phi.dimensions, internal=True)
    else:
        return Field('div({0})'.format(phi.name), divField, phi.dimensions)

def grad(phi, ghost=False, transpose=False):
    assert len(phi.dimensions) == 1
    logger.info('gradient of {0}'.format(phi.name))
    mesh = phi.mesh
    if phi.dimensions[0] == 1:
        #WTF is this needed?
        phi.field = phi.field.reshape((-1, 1))
        product = phi * mesh.Normals
        dimensions = (3,)
    else:
        if transpose:
            product = phi.outer(mesh.Normals)
        else:
            product = mesh.Normals.outer(phi)
        product.field = product.field.reshape((mesh.nFaces, 9))
        dimensions = (3,3)
    gradField = internal_sum(product)
    # if grad of scalar
    if phi.dimensions[0] == 3:
        gradField = gradField.reshape((mesh.nInternalCells, 3, 3))
    if ghost:
        return CellField('grad({0})'.format(phi.name), gradField, dimensions, internal=True)
    else:
        return Field('grad({0})'.format(phi.name), gradField, dimensions)

def snGrad(phi):
    logger.info('snGrad of {0}'.format(phi.name))
    mesh = phi.mesh
    gradFdotn = (phi.field[mesh.neighbour]-phi.field[mesh.owner])/mesh.deltas
    return Field('snGrad({0})'.format(phi.name), gradFdotn, phi.dimensions)

def laplacian(phi, DT):
    logger.info('laplacian of {0}'.format(phi.name))
    mesh = phi.mesh

    # non orthogonal correction
    #DTgradF = Field.zeros('grad' + phi.name, mesh, mesh.nCells, 3.)
    #DTgradF.setInternalField(DT*grad(field))
    #laplacian1 = div(interpolate(DTgradF), 1.)

    gradFdotn = snGrad(phi)
    laplacian2 = internal_sum(gradFdotn*DT)
    return Field('laplacian({0})'.format(phi.name), laplacian2, phi.dimensions)

def ddt(phi, dt):
    logger.info('ddt of {0}'.format(phi.name))
    return Field('ddt' + phi.name, (phi.getInternalField()-phi.getInternalField())/dt, phi.dimensions)
    #return Field('ddt' + phi.name, (phi.getInternalField()-phi.old.getInternalField())/dt, phi.dimensions)


