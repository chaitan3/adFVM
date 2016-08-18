from __future__ import print_function
from test import *

from adFVM.field import Field
from adFVM.mesh import Mesh
from adFVM.op import grad, div, laplacian, snGrad

class TestOp(TestAdFVM):
    def test_grad_scalar(self):

        ref = np.zeros((self.meshO.nInternalCells, 3))
        X = self.X[:self.meshO.nInternalCells]
        Y = self.Y[:self.meshO.nInternalCells]
        ref[:, 0] = Y + 2*X + 1
        ref[:, 1] = X + 2*Y

        T = np.zeros((self.meshO.nFaces, 1))
        T[:,0] = self.XF*self.YF + self.XF**2 + self.YF**2 + self.XF
        R = grad(self.FV)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))
        res = evaluate(R.field, self.V, T, self)
        checkVolSum(self, res, ref)

        T = np.zeros((self.meshO.nCells, 1))
        T[:,0] = self.X*self.Y + self.X**2 + self.Y**2 + self.X
        R = grad(self.FV, op=True)
        res = evaluate(R.field, self.V, T, self)
        checkVolSum(self, res, ref)
        
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
        unittest.main(verbosity=2, buffer=True)
