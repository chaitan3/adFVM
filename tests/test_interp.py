from adFVM.interp import central, centralOld, secondOrder
from adFVM.op import grad
from adFVM.mesh import Mesh
from adFVM.field import Field
from adpy.variable import Variable, Function, Zeros
from adpy.tensor import Kernel

import numpy as np

def test_second_order(self):
    Field.setSolver(self)
    R = SecondOrder(self)
    GR = grad(self.FU, op=True, ghost=True)
    R = R.dual(self.FU, GR)
    ref = np.zeros((self.meshO.nFaces, 3))
    ref[:,0] = np.sin(2*self.XF*np.pi)*np.sin(2*self.YF*np.pi)
    ref[:,1] = np.sin(2*self.YF*np.pi)
    
    T = np.zeros((self.meshO.nCells, 3))
    T[:,0] = np.sin(2*self.X*np.pi)*np.sin(2*self.Y*np.pi)
    T[:,1] = np.sin(2*self.Y*np.pi)
    res = evaluate([x.field for x in R], self.U, T, self)
    m = self.meshO.nInternalCells
    n = self.meshO.nInternalFaces
    res2 = evaluate(GR.field, self.U, T, self)
    checkArray(self, res[0], ref, maxThres=1e-3)
    checkArray(self, res[1], ref, maxThres=1e-3)

def test_central():
    case = '../cases/convection'
    mesh = Mesh.create(case)
    Field.setMesh(mesh)
    thres = 1e-12

    X, Y = mesh.cellCentres[:, [0]], mesh.cellCentres[:,[1]]
    Uc = Field('U', 2*X + X*Y, (1,))
    XF, YF = mesh.faceCentres[:, [0]], mesh.faceCentres[:,[1]]
    Ur = 2*XF + XF*YF
    Uf = centralOld(Uc, mesh)
    assert np.max(np.abs(Ur-Uf.field))/np.max(np.abs(Ur)) < thres

    U = Variable((mesh.symMesh.nInternalCells, 1))
    def interpolate(U, *meshArgs):
        mesh = Mesh.container(meshArgs)
        return central(U, mesh)
    Uf = Zeros((mesh.symMesh.nFaces, 1))
    meshArgs = mesh.symMesh.getTensor()
    Uf = Kernel(interpolate)(mesh.symMesh.nFaces, (Uf,))(U, *meshArgs)[0]
    meshArgs = mesh.symMesh.getTensor() + mesh.symMesh.getScalar()
    func = Function('interpolate', [U] + meshArgs, (Uf,))

    Function.compile()
    meshArgs = mesh.getTensor() + mesh.getScalar()
    Uf = func(Uc.field, *meshArgs)[0]
    assert np.max(np.abs(Ur-Uf))/np.max(np.abs(Ur)) < thres

if __name__ == "__main__":
    test_central()
