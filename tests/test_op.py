from __future__ import print_function

from adFVM.field import Field
from adFVM.mesh import Mesh
from adFVM.op import grad, gradCell, div, laplacian, snGrad
from adFVM import config

from adpy.variable import Variable, Function, Zeros
from adpy.tensor import Kernel

import numpy as np

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
    gradU = Kernel(_grad)(mesh.symMesh.nInternalCells, (gradU,))(U, *meshArgs)[0]
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('grad_scalar', [U] + meshArgs, (gradU,))

    Function.compile(init=False, compiler_args=config.get_compiler_args())
    Function.initialize(0, mesh)

    meshArgs = mesh.getTensor() + mesh.getScalar()
    #gradU = func(Uf, *meshArgs)[0]
    gradU = func(Uc, *meshArgs)[0]
    assert relative_error(gradU, gradUr[:mesh.nInternalCells]) < thres

def test_grad_vector(self):
    ref = np.zeros((self.meshO.nInternalCells, 3, 3))
    X = self.X[:self.meshO.nInternalCells]
    Y = self.Y[:self.meshO.nInternalCells]
    ref[:, 0, 0] = Y + 2*X
    ref[:, 0, 1] = X
    ref[:, 1, 1] = 1 + 2*Y

    T = np.zeros((self.meshO.nFaces, 3))
    T[:, 0] = self.XF*self.YF + self.XF**2
    T[:, 1] = self.YF + self.YF**2 
    T[:, 2] = 1.
    R = grad(self.FU)
    res = evaluate(R.field, self.U, T, self)
    self.assertTrue(isinstance(R, Field))
    self.assertEqual(R.dimensions, (3,3))
    checkVolSum(self, res, ref)

    T = np.zeros((self.meshO.nCells, 3))
    T[:, 0] = self.X*self.Y + self.X**2
    T[:, 1] = self.Y + self.Y**2 
    T[:, 2] = 1.
    R = grad(self.FU, op=True)
    res = evaluate(R.field, self.U, T, self)
    checkVolSum(self, res, ref)

def test_div(self):
    R = div(self.FU.dotN())
    self.assertTrue(isinstance(R, Field))
    self.assertEqual(R.dimensions, (1,))

    T = np.zeros((self.meshO.nFaces, 3))
    T[:, 0] = self.XF + np.sin(2*np.pi*self.XF)*np.cos(2*np.pi*self.YF)
    T[:, 1] = self.YF**2 - np.cos(2*np.pi*self.XF)*np.sin(2*np.pi*self.YF)
    T[:, 2] = self.XF
    res = evaluate(R.field, self.U, T, self)
    Y = self.Y[:self.meshO.nInternalCells]
    ref = (1 + 2*Y).reshape(-1,1)
    checkVolSum(self, res, ref)

def test_laplacian(self):
    R = laplacian(self.FV, 1.)
    self.assertTrue(isinstance(R, Field))
    self.assertEqual(R.dimensions, (1,))

    T = np.zeros((self.meshO.nCells, 1))
    T[:, 0] = self.X**2 + self.Y**2 + self.X*self.Y
    res = evaluate(R.field, self.V, T, self)
    ref = 4.*np.ones((self.meshO.nInternalCells, 1))
    checkVolSum(self, res, ref, relThres=1e-2)

def test_snGrad(self):
    ref = np.zeros((self.meshO.nFaces, 1))
    X = self.XF
    Y = self.YF
    N = self.meshO.normals
    nx, ny = N[:,0], N[:,1]
    ref[:, 0] = (Y + 2*X + 1)*nx + (X + 2*Y)*ny

    T = np.zeros((self.meshO.nCells, 1))
    T[:,0] = self.X*self.Y + self.X**2 + self.Y**2 + self.X
    R = snGrad(self.FV)
    self.assertTrue(isinstance(R, Field))
    self.assertEqual(R.dimensions, (1,))
    res = evaluate(R.field, self.V, T, self)
    checkArray(self, res, ref)

if __name__ == "__main__":
    test_grad_scalar()
