from __future__ import print_function
import numpy as np

from . import config
from .field import Field, CellField, IOField

from adpy.tensor import Tensor

logger = config.Logger(__name__)


def div(phi, mesh, neighbour):
    # needs to be already dotted with mesh normals
    wp = mesh.areas/mesh.volumesL
    if neighbour:
        wn = -mesh.areas/mesh.volumesR
        dphi = Tensor.collate(phi*wp, mesh.owner, phi*wn, mesh.neighbour)
    else:
        dphi = Tensor.collate(phi*wp, mesh.owner)
    return dphi

def absDiv(phi, mesh, neighbour):
    wp = mesh.areas/mesh.volumesL
    if neighbour:
        wn = mesh.areas/mesh.volumesR
        dphi = Tensor.collate(phi*wp, mesh.owner, phi*wn, mesh.neighbour)
    else:
        dphi = Tensor.collate(phi*wp, mesh.owner)
    return dphi


def grad(phi, mesh, neighbour):
    wp = mesh.areas/mesh.volumesL
    if phi.shape == (1,):
        phiN = phi*mesh.normals
    else:
        phiN = phi.outer(mesh.normals)
    if neighbour:
        wn = -mesh.areas/mesh.volumesR
        gphi = Tensor.collate(phiN*wp, mesh.owner, phiN*wn, mesh.neighbour)
    else:
        gphi = Tensor.collate(phiN*wp, mesh.owner)
    return gphi

def gradCell(phi, mesh):
    gradPhi = 0
    nCellFaces = 6
    for i in range(0, nCellFaces):
        P = mesh.cellFaces[i]
        S = mesh.areas.extract(P)
        N = mesh.normals.extract(P)
        w = mesh.weights.extract(P) 
        O = mesh.cellOwner[i]
        N = 2*N*O-N
        w = w + O - 2*w*O
        phiP = phi.extract(mesh.cellNeighbours[i])
        phiF = phi.index()*(-w + 1) + phiP*w
        if phi.shape == (1,):
            gradPhi += phiF*S*N
        else:
            gradPhi += phiF.outer(S*N)
    gradPhi = gradPhi/mesh.volumes
    return gradPhi


def snGrad(phiL, phiR, mesh):
    return (phiR - phiL)/mesh.deltas

def snGradCorr(phiL, phiR, gradPhiF, mesh):
    implicit = (phiR - phiL)/mesh.deltas
    cost = mesh.deltasUnit.dot(mesh.normals)
    if phiL.shape == (1,):
        explicit = gradPhiF[0].dot(mesh.normals-mesh.deltasUnit/cost)
    else:
        explicit = gradPhiF.tensordot(mesh.normals-mesh.deltasUnit/cost)
    return implicit/cost + explicit
   

# code gen ends heere
def internal_sum(phi, mesh, sumOp=False, absolute=False):
    if sumOp:
        if not absolute:
            sumOp = mesh.sumOp
        else:
            sumOp = np.abs(mesh.sumOp)
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

    return x


def divOld(phi, U=None, ghost=False):
    logger.info('divergence of {0}'.format(phi.name))
    mesh = phi.mesh
    if U is None:
        divField = internal_sum(phi, mesh)
    else:
        assert phi.dimensions == (1,)
        divField = internal_sum((phi*U).dotN(), mesh)
    if ghost:
        divPhi = CellField('div({0})'.format(phi.name), divField, phi.dimensions, ghost=True)
        return divPhi
    else:
        return Field('div({0})'.format(phi.name), divField, phi.dimensions)

def gradOld(phi, op=False, sumOp=False, ghost=False):
    assert len(phi.dimensions) == 1
    logger.info('gradient of {0}'.format(phi.name))
    mesh = phi.mesh
    if phi.dimensions == (1,):
        dimensions = (3,)
    else:
        dimensions = phi.dimensions + (3,)

    if op:
        gradField = adsparse.basic.dot(mesh.gradOp, phi.field)
        gradField = gradField.reshape((mesh.nInternalCells,) + dimensions)
        if dimensions == (3,3):
            gradField = gradField.transpose((0, 2, 1))
    else:
        N = Field('N', mesh.normals, (3,))
        if dimensions == (3,):
            product = phi * N
        else:
            product = phi.outer(N)
            dimprod = np.prod(dimensions)
            product.field = np.reshape(product.field, (mesh.nFaces, dimprod))
        gradField = internal_sum(product, mesh, sumOp=sumOp)
        # if grad of vector
        if len(dimensions) == 2:
            gradField = np.reshape(gradField, (mesh.nInternalCells,) + dimensions)

    return IOField('grad({0})'.format(phi.name), gradField, dimensions, ghost=ghost)

