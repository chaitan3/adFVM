from __future__ import print_function
import numpy as np
import time
import scipy.sparse as sp

from field import Field, CellField
from interp import interpolate
from config import ad, Logger
logger = Logger(__name__)

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
    if phi.dimensions[0] == 1:
        product = phi * mesh.Normals
    else:
        product = mesh.Normals.outer(phi)
        product.field = product.field.reshape((phi.size, 9))
    gradField = (mesh.sumOp * (product.field * mesh.areas))/mesh.volumes
    # if grad of scalar
    if phi.dimensions[0] == 3:
        gradField = gradField.reshape((mesh.nInternalCells, 3, 3))
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

    gradFdotn = snGrad(phi)
    laplacian2 = (mesh.sumOp * ((DT * gradFdotn).field * mesh.areas))/mesh.volumes
    return Field('laplacian({0})'.format(phi.name), mesh, laplacian2)

def ddt(phi, dt):
    logger.info('ddt of {0}'.format(phi.name))
    return Field('ddt' + phi.name, phi.mesh, (phi.getInternalField()-phi.old.getInternalField())/dt)


