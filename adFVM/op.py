from __future__ import print_function
import numpy as np

from . import config
from .field import Field, CellField, IOField

logger = config.Logger(__name__)


def internal_sum(phi, mesh, absolute=False):
    if 0:#config.device == "cpu":
        if not absolute:
            sumOp = mesh.sumOp
        else:
            sumOp = ad.abs(mesh.sumOp)
        #x = (adsparse.basic.dot(sumOp, (phi.field * mesh.areas)))/mesh.volumes
        x = ad.sparse_tensor_dense_matmul(sumOp, phi.field * mesh.areas)/mesh.volumes
    else:
        phiF = phi.field*mesh.areas
        dimensions = (np.product(phi.dimensions),)
        x = np.zeros((mesh.nInternalCells+1,) + dimensions, config.precision)
        np.add.at(x, mesh.owner, phiF)
        if not absolute:
            np.add.at(x, mesh.neighbour[:mesh.nInternalFaces], -phiF[:mesh.nInternalFaces])
        else:
            np.add.at(x, mesh.neighbour[:mesh.nInternalFaces], phiF[:mesh.nInternalFaces])
        x = x[:-1]/mesh.volumes

    # retain pattern broadcasting
    #x = ad.patternbroadcast(x, phi.field.broadcastable)
    return x


def internal_sum_numpy(phi, mesh):
    return (mesh.sumOp * (phi.field * mesh.areas))/mesh.volumes


def div(phi, mesh):
    wp = mesh.areas/mesh.volumesL
    wn = mesh.areas/mesh.volumesR
    # for div, contri for owner is pos, neigh is neg
    return phi*wp, phi*wn

#def div(phi, U=None, ghost=False):
#    logger.info('divergence of {0}'.format(phi.name))
    #mesh = phi.mesh
    #if U is None:
    #    divField = internal_sum(phi, mesh)
    #else:
    #    assert phi.dimensions == (1,)
    #    divField = internal_sum((phi*U).dotN(), mesh)
    #if ghost:
    #    divPhi = CellField('div({0})'.format(phi.name), divField, phi.dimensions, ghost=True)
    #    return divPhi
    #else:
    #    return Field('div({0})'.format(phi.name), divField, phi.dimensions)

def snGrad(phiL, phiR, mesh):
    return (phiR - phiL)/mesh.deltas

def laplacian(phi, DT):
    logger.info('laplacian of {0}'.format(phi.name))
    mesh = phi.mesh
    gradFdotn = snGrad(phi)
    laplacian2 = internal_sum(gradFdotn*DT, mesh)
    return Field('laplacian({0})'.format(phi.name), laplacian2, phi.dimensions)

# dual defined 
def grad(phi, ghost=False, op=False, numpy=False):
    assert len(phi.dimensions) == 1
    logger.info('gradient of {0}'.format(phi.name))
    mesh = phi.mesh
    if phi.dimensions == (1,):
        dimensions = (3,)
    else:
        dimensions = phi.dimensions + (3,)

    if numpy:
        assert not op
        loc_internal_sum = internal_sum_numpy
        mod = IOField
        mesh = mesh.origMesh
        ad = np
    else:
        loc_internal_sum = internal_sum
        mod = CellField

    if op and config.device == 'cpu':
        gradField = adsparse.basic.dot(mesh.gradOp, phi.field)
        gradField = gradField.reshape((mesh.nInternalCells,) + dimensions)
        if dimensions == (3,3):
            gradField = gradField.transpose((0, 2, 1))
    else:
        if dimensions == (3,):
            product = phi * mesh.Normals
        else:
            product = phi.outer(mesh.Normals)
            dimprod = np.prod(dimensions)
            product.field = ad.reshape(product.field, (mesh.nFaces, dimprod))
        gradField = loc_internal_sum(product, mesh)
        # if grad of vector
        if len(dimensions) == 2:
            gradField = ad.reshape(gradField, (mesh.nInternalCells,) + dimensions)

    if ghost:
        gradPhi = mod('grad({0})'.format(phi.name), gradField, dimensions, ghost=True)
        return gradPhi
    else:
        return Field('grad({0})'.format(phi.name), gradField, dimensions)



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


