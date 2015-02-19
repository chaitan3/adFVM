from __future__ import print_function
import numpy as np
import time

import config
from config import ad, adsparse
from field import Field, CellField

logger = config.Logger(__name__)

def internal_sum(phi, mesh):
    x = (adsparse.basic.dot(mesh.sumOp, (phi.field * mesh.areas)))/mesh.volumes
    # retain pattern broadcasting
    x = ad.patternbroadcast(x, phi.field.broadcastable)
    return x

def div(phi, U=None, ghost=False):
    logger.info('divergence of {0}'.format(phi.name))
    mesh = phi.mesh
    if U is None:
        divField = internal_sum(phi, mesh)
    else:
        assert phi.dimensions == (1,)
        raise Exception('not tested')
        divField = internal_sum((phi*U).dotN(), mesh)
    if ghost:
        return CellField('div({0})'.format(phi.name), divField, phi.dimensions, internal=True)
    else:
        return Field('div({0})'.format(phi.name), divField, phi.dimensions)

def grad(phi, ghost=False):
    assert len(phi.dimensions) == 1
    logger.info('gradient of {0}'.format(phi.name))
    if ghost:
        mesh = phi.mesh.paddedMesh
    else:
        mesh = phi.mesh
    if phi.dimensions[0] == 1:
        product = phi * mesh.Normals
        dimensions = (3,)
    else:
        product = phi.outer(mesh.Normals)
        product.field = product.field.reshape((mesh.nFaces, 9))
        dimensions = (3,3)
    gradField = internal_sum(product, mesh)
    # if grad of scalar
    if phi.dimensions[0] == 3:
        gradField = gradField.reshape((mesh.nInternalCells, 3, 3))
    if ghost:
        gradPhi = CellField('grad({0})'.format(phi.name), gradField, dimensions, ghost=True)
        gradPhi.copyRemoteCells(gradField)
        return gradPhi
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
    laplacian2 = internal_sum(gradFdotn*DT, mesh)
    return Field('laplacian({0})'.format(phi.name), laplacian2, phi.dimensions)

def ddt(phi, dt):
    logger.info('ddt of {0}'.format(phi.name))
    #return Field('ddt' + phi.name, (phi.getInternalField()-phi.getInternalField())/dt, phi.dimensions)
    return Field('ddt' + phi.name, phi.getInternalField()*0, phi.dimensions)
    #return Field('ddt' + phi.name, (phi.getInternalField()-phi.old.getInternalField())/dt, phi.dimensions)


