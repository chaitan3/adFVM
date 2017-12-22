from adFVM.interp import central, centralOld, secondOrder
from adFVM.op import gradOld
from adFVM.mesh import Mesh
from adFVM.field import Field
from adFVM import config

from adpy.variable import Variable, Function, Zeros
from adpy.tensor import Kernel

import numpy as np
import pytest

@pytest.mark.skip
def relative_error(U, Ur):
    return np.max(np.abs(U-Ur))/np.max(np.abs(Ur))

def test_second_order():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-5

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Uc = Field('U', 2*X + X**3*Y**2, (1,))
    gradUc = gradOld(centralOld(Uc, mesh), ghost=True)
    gradUc.field = gradUc.field.reshape((mesh.nCells, 1, 3))
    Xf, Yf = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    Ur = 2*Xf + Xf**3*Yf**2

    U = Variable((mesh.symMesh.nCells, 1))
    gradU = Variable((mesh.symMesh.nCells, 1, 3))
    def interpolate(U, gradU, *meshArgs):
        mesh = Mesh.container(meshArgs)
        return secondOrder(U, gradU, mesh, 0)
    Uf = Zeros((mesh.symMesh.nFaces, 1))
    meshArgs = mesh.symMesh.getTensor()
    Uf = Kernel(interpolate)(mesh.symMesh.nFaces, (Uf,))(U, gradU, *meshArgs)
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('second_order', [U, gradU] + meshArgs, (Uf,))

    Function.compile(init=False, compiler_args=config.get_compiler_args())
    Function.initialize(0, mesh)

    meshArgs = mesh.getTensor() + mesh.getScalar()
    Uf = func(Uc.field, gradUc.field, *meshArgs)[0]
    assert relative_error(Uf, Ur) < thres

def test_central():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-12

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Uc = Field('U', 2*X + X*Y, (1,))
    Xf, Yf = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    Ur = 2*Xf + Xf*Yf
    Uf = centralOld(Uc, mesh).field
    assert relative_error(Uf, Ur) < thres

    U = Variable((mesh.symMesh.nInternalCells, 1))
    def interpolate(U, *meshArgs):
        mesh = Mesh.container(meshArgs)
        return central(U, mesh)
    Uf = Zeros((mesh.symMesh.nFaces, 1))
    meshArgs = mesh.symMesh.getTensor()
    Uf = Kernel(interpolate)(mesh.symMesh.nFaces, (Uf,))(U, *meshArgs)
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('central', [U] + meshArgs, (Uf,))

    Function.compile()
    meshArgs = mesh.getTensor() + mesh.getScalar()
    Uf = func(Uc.field, *meshArgs)[0]
    assert relative_error(Uf, Ur) < thres

if __name__ == "__main__":
    #test_central()
    test_second_order()
