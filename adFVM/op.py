from __future__ import print_function
import numpy as np
import time

from . import config
from .config import ad, adsparse
from .field import Field, CellField

logger = config.Logger(__name__)

def internal_sum(phi, mesh, absolute=False):
    if config.device == "cpu":
        if not absolute:
            sumOp = mesh.sumOp
        else:
            sumOp = adsparse.basic.sp_ones_like(mesh.sumOp)
        x = (adsparse.basic.dot(sumOp, (phi.field * mesh.areas)))/mesh.volumes
    else:
        phiF = phi.field*mesh.areas
        dimensions = (np.product(phi.dimensions),)
        x = ad.bcalloc(config.precision(0.), (mesh.nInternalCells+1,) + dimensions)
        x = ad.inc_subtensor(x[mesh.owner], phiF)
        if not absolute:
            x = ad.inc_subtensor(x[mesh.neighbour[:mesh.nInternalFaces]], -phiF[:mesh.nInternalFaces])
        else:
            x = ad.inc_subtensor(x[mesh.neighbour[:mesh.nInternalFaces]], phiF[:mesh.nInternalFaces])
        # TODO: mesh.repeat disabled, won't work correctly in parallel
        #x = ad.set_subtensor(x[mesh.repeat[0]], x[mesh.repeat].sum(axis=0))
        x = x[:-1]/mesh.volumes

    # retain pattern broadcasting
    x = ad.patternbroadcast(x, phi.field.broadcastable)
    return x

def div(phi, U=None, ghost=False):
    logger.info('divergence of {0}'.format(phi.name))
    mesh = phi.mesh
    if U is None:
        divField = internal_sum(phi, mesh)
    else:
        raise Exception('not tested')
        assert phi.dimensions == (1,)
        divField = internal_sum((phi*U).dotN(), mesh)
    if ghost:
        divPhi = CellField('div({0})'.format(phi.name), divField, phi.dimensions, ghost=True)
        return divPhi
    else:
        return Field('div({0})'.format(phi.name), divField, phi.dimensions)

def grad(phi, ghost=False):
    assert len(phi.dimensions) == 1
    logger.info('gradient of {0}'.format(phi.name))
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
    gradFdotn = snGrad(phi)
    laplacian2 = internal_sum(gradFdotn*DT, mesh)
    return Field('laplacian({0})'.format(phi.name), laplacian2, phi.dimensions)

# only defined for ndarray
def curl(phi):
    assert phi.dimensions == (3,)
    assert isinstance(phi.field, np.ndarray)
    logger.info('vorticity of {0}'.format(phi.name))
    mesh = phi.mesh
    vort = Field('N', mesh.origMesh.normals, (3,)).cross(phi)
    vort.name = 'curl({0})'.format(phi.name)
    return vort

def ddt(phi, dt):
    logger.info('ddt of {0}'.format(phi.name))
    raise Exception("Deprecated")
    #return Field('ddt' + phi.name, (phi.getInternalField()-phi.getInternalField())/dt, phi.dimensions)
    return Field('ddt' + phi.name, phi.getInternalField()*0, phi.dimensions)
    #return Field('ddt' + phi.name, (phi.getInternalField()-phi.old.getInternalField())/dt, phi.dimensions)


