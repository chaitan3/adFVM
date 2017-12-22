from __future__ import print_function

from adFVM.field import Field
from adFVM.mesh import Mesh
from adFVM.op import grad, gradCell, div, snGrad
from adFVM import config

from adpy.variable import Variable, Function, Zeros
from adpy.tensor import Kernel

import numpy as np
import pytest

@pytest.mark.skip
def relative_error(U, Ur):
    return np.max(np.abs(U-Ur))/np.max(np.abs(Ur))

def test_grad_scalar():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-9

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Xf, Yf = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    #Uf = Xf*Yf + Xf**2 + Yf**2 + Xf
    Uc = X*Y + X**2 + Y**2 + X
    gradUr = np.concatenate((
            Y + 2*X + 1,
            X + 2*Y,
            X*0
        ), axis=1).reshape((-1, 1, 3))

    U = Variable((mesh.symMesh.nCells, 1))
    def _grad(U, *meshArgs):
        mesh = Mesh.container(meshArgs)
        #return grad(U, mesh)
        return gradCell(U, mesh)
    gradU = Zeros((mesh.symMesh.nInternalCells, 1, 3))
    meshArgs = mesh.symMesh.getTensor()
    gradU = Kernel(_grad)(mesh.symMesh.nInternalCells, (gradU,))(U, *meshArgs)
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('grad_scalar', [U] + meshArgs, (gradU,))

    Function.compile(init=False, compiler_args=config.get_compiler_args())
    Function.initialize(0, mesh)

    meshArgs = mesh.getTensor() + mesh.getScalar()
    #gradU = func(Uf, *meshArgs)[0]
    gradU = func(Uc, *meshArgs)[0]
    assert relative_error(gradU, gradUr[:mesh.nInternalCells]) < thres

def test_grad_vector():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-9

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Xf, Yf = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    #Uf = Xf*Yf + Xf**2 + Yf**2 + Xf
    Uc = np.concatenate((
            X*Y + X**2,
            Y + Y**2,
            np.ones_like(X)
        ), axis=1)
    gradUr = np.zeros((mesh.nCells, 3, 3))
    gradUr[:, 0, 0] = (Y + 2*X).flatten()
    gradUr[:, 0, 1] = (X).flatten()
    gradUr[:, 1, 1] = (1 + 2*Y).flatten()

    U = Variable((mesh.symMesh.nCells, 3))
    def _grad(U, *meshArgs):
        mesh = Mesh.container(meshArgs)
        #return grad(U, mesh)
        return gradCell(U, mesh)
    gradU = Zeros((mesh.symMesh.nInternalCells, 3, 3))
    meshArgs = mesh.symMesh.getTensor()
    gradU = Kernel(_grad)(mesh.symMesh.nInternalCells, (gradU,))(U, *meshArgs)
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('grad_vector', [U] + meshArgs, (gradU,))

    Function.compile(init=False, compiler_args=config.get_compiler_args())
    Function.initialize(0, mesh)

    meshArgs = mesh.getTensor() + mesh.getScalar()
    #gradU = func(Uf, *meshArgs)[0]
    gradU = func(Uc, *meshArgs)[0]
    assert relative_error(gradU, gradUr[:mesh.nInternalCells]) < thres

def test_div():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-9

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Xf, Yf = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    Uf = np.concatenate((
        Xf + np.sin(2*np.pi*Xf)*np.cos(2*np.pi*Yf),
        Yf**2 - np.cos(2*np.pi*Xf)*np.sin(2*np.pi*Yf),
        Xf
        ), axis=1)
    Uf = (Uf*mesh.normals).sum(axis=1)

    divUr = (1 + 2*Y)

    U = Variable((mesh.symMesh.nFaces, 1))
    def _meshArgs(start=0):
        return [x[start] for x in mesh.symMesh.getTensor()]
    def _div(U, *meshArgs, **options):
        neighbour = options.pop('neighbour', True)
        mesh = Mesh.container(meshArgs)
        #return grad(U, mesh)
        return div(U, mesh, neighbour)
    divU = Zeros((mesh.symMesh.nInternalCells, 1))
    meshArgs = _meshArgs()
    divU = Kernel(_div)(mesh.symMesh.nInternalFaces, (divU,))(U, neighbour=True, *meshArgs)
    meshArgs = _meshArgs(mesh.symMesh.nInternalFaces)
    divU = Kernel(_div)(mesh.symMesh.nGhostCells, (divU,))(U[mesh.symMesh.nInternalFaces], *meshArgs, neighbour=False)
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('div', [U] + meshArgs, (divU,))

    Function.compile(init=False, compiler_args=config.get_compiler_args())
    Function.initialize(0, mesh)

    meshArgs = mesh.getTensor() + mesh.getScalar()
    divU = func(Uf, *meshArgs)[0]
    assert relative_error(divU, divUr[:mesh.nInternalCells]) < thres

def test_snGrad():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-9

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Xf, Yf = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    nx, ny = mesh.normals[:,[0]], mesh.normals[:,[1]]
    Uc = X*Y + X**2 + Y**2 + X
    snGradUr = (Yf + 2*Xf + 1)*nx + (Xf + 2*Yf)*ny

    U = Variable((mesh.symMesh.nCells, 1))
    def _meshArgs(start=0):
        return [x[start] for x in mesh.symMesh.getTensor()]
    def _snGrad(U, *meshArgs, **options):
        mesh = Mesh.container(meshArgs)
        return snGrad(U.extract(mesh.owner), U.extract(mesh.neighbour), mesh)
    snGradU = Zeros((mesh.symMesh.nFaces, 1))
    meshArgs = _meshArgs()
    snGradU = Kernel(_snGrad)(mesh.symMesh.nFaces, (snGradU,))(U, *meshArgs)
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('snGrad', [U] + meshArgs, (snGradU,))

    Function.compile(init=False, compiler_args=config.get_compiler_args())
    Function.initialize(0, mesh)

    meshArgs = mesh.getTensor() + mesh.getScalar()
    snGradU = func(Uc, *meshArgs)[0]
    assert relative_error(snGradU, snGradUr) < thres


if __name__ == "__main__":
    #test_grad_scalar()
    #test_grad_vector()
    #test_div()
    test_snGrad()
