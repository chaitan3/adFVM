from __future__ import print_function
from test import *

from adFVM import config
from adFVM.field import Field, CellField
from adFVM.mesh import Mesh

class TestField(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = '../cases/convection/'
        self.mesh = Mesh.create(self.case)
        Field.setSolver(self)

        self.meshO = self.mesh.origMesh
        self.X = self.meshO.cellCentres[:self.meshO.nInternalCells, 0]
        self.Y = self.meshO.cellCentres[:self.meshO.nInternalCells, 1]

        self.U = ad.matrix()
        self.FU = Field('F', self.U, (3,))
        self.V = ad.matrix()
        self.FV = Field('F', self.V, (3,))
        self.W = ad.tensor3()
        self.FW = Field('F', self.W, (3,3))

        #self.U.field[:, 0] = self.X
        #self.U.field[:, 1] = self.Y
        #self.U.field[:, 2] = 0.5
        #self.V.field[:, 0] = self.Y
        #self.V.field[:, 1] = -self.X
        #self.V.field[:, 2] = 2.
        #self.W.field[:, 0] = self.X*0.1
        #self.W.field[:, 1] = self.X*0.2
        #self.W.field[:, 2] = self.X*0.3
        #self.T.field[:, 0, 0] = self.X*self.X
        #self.T.field[:, 0, 1] = self.X*self.Y
        #self.T.field[:, 1, 1] = self.Y*self.Y
        #self.T.field[:, 2, 2] = 1.

    def test_max(self):
        FU = Field('F', self.U, (1,))
        FV = Field('F', self.V, (1,))
        R = Field.max(FU, FV)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        S = np.random.rand(self.meshO.nFaces, 1) 
        T = np.random.rand(self.meshO.nFaces, 1)
        res = evaluate(R.field, [self.U, self.V], [S,T], self)
        ref = np.maximum(S, T)
        checkArray(self, res, ref)

    def test_switch(self):
        C = ad.matrix()
        FU = Field('F', self.U, (1,))
        FV = Field('F', self.V, (1,))
        R = Field.switch(C, FU, FV)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        S = np.random.rand(self.meshO.nFaces, 1)
        T = np.random.rand(self.meshO.nFaces, 1)
        P = S > T
        res = evaluate(R.field, [C, self.U, self.V], [P,S,T], self)
        ref = np.maximum(S, T)
        checkArray(self, res, ref)

    def test_abs(self):
        R = self.FU.abs()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        T = np.zeros((self.meshO.nInternalCells, 3))
        T[:,0] = self.X
        T[:,1] = -1
        T[:,2] = 2.
        res = evaluate(R.field, self.U, T, self)
        ref = np.zeros_like(T)
        ref[:,0] = np.abs(self.X)
        ref[:,1] = 1.
        ref[:,2] = 2.
        checkArray(self, res, ref)

    def test_sign(self):
        R = self.FU.sign()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        T = np.zeros((self.meshO.nInternalCells, 3))
        T[:,0] = self.X
        T[:,1] = -1
        T[:,2] = 2.
        res = evaluate(R.field, self.U, T, self)
        ref = np.zeros_like(T)
        ref[:,0] = np.sign(self.X)
        ref[:,1] = -1.
        ref[:,2] = 1.
        checkArray(self, res, ref)

    def test_component(self):
        R = self.FU.component(0)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        T = np.random.rand(self.meshO.nFaces, 3)
        res = evaluate(R.field, self.U, T, self)
        ref = T[:,[0]]
        checkArray(self, res, ref)

    def test_magSqr(self):
        R = self.FU.magSqr()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        T = np.zeros((self.meshO.nInternalCells, 3))
        T[:, 0] = self.X
        T[:, 1] = self.Y
        res = evaluate(R.field, self.U, T, self)
        ref = (self.X**2 + self.Y**2).reshape(-1,1)
        checkArray(self, res, ref)

    def test_dot_vector(self):
        R = self.FU.dot(self.FV)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        S = np.zeros((self.meshO.nInternalCells, 3))
        T = np.zeros((self.meshO.nInternalCells, 3))
        S[:,0], S[:,1] = self.X, self.Y
        T[:,0], T[:,1] = self.Y, -self.X
        res = evaluate(R.field, [self.U, self.V], [S, T], self)
        ref = np.zeros((self.meshO.nInternalCells, 1))
        checkArray(self, res, ref)

    def test_outer(self):
        R = self.FU.outer(self.FV)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,3))

        S = np.zeros((self.meshO.nInternalCells, 3))
        T = np.zeros((self.meshO.nInternalCells, 3))
        S[:,0], S[:,1], S[:,2] = self.X, self.Y, 0.5
        T[:,0], T[:,1], T[:,2] = self.Y, -self.X, 2.
        res = evaluate(R.field, [self.U, self.V], [S, T], self)
        ref = np.zeros((self.meshO.nInternalCells, 3, 3))
        ref[:, 0, 0] = self.X*self.Y
        ref[:, 0, 1] = -self.X*self.X
        ref[:, 0, 2] = 2*self.X
        ref[:, 1, 0] = self.Y*self.Y
        ref[:, 1, 1] = -self.X*self.Y
        ref[:, 1, 2] = 2*self.Y
        ref[:, 2, 0] = 0.5*self.Y
        ref[:, 2, 1] = -0.5*self.X
        ref[:, 2, 2] = 1.
        checkArray(self, res, ref)

    def test_dot_tensor(self):
        R = self.FW.dot(self.FU)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        S = np.zeros((self.meshO.nInternalCells, 3, 3))
        T = np.zeros((self.meshO.nInternalCells, 3))
        S[:, 0, 0] = self.X*self.X
        S[:, 0, 1] = self.X*self.Y
        S[:, 1, 1] = self.Y*self.Y
        S[:, 2, 2] = 1.
        T[:,0], T[:,1], T[:,2] = self.X*0.1, self.X*0.2, self.X*0.3
        res = evaluate(R.field, [self.W, self.U], [S, T], self)
        ref = np.zeros((self.meshO.nInternalCells, 3))
        ref[:, 0] = (S[:, 0, 0]*0.1 + S[:, 0, 1]*0.2)*self.X
        ref[:, 1] = S[:, 1, 1]*0.2*self.X
        ref[:, 2] = S[:, 2, 2]*0.3*self.X
        checkArray(self, res, ref)

    def test_transpose(self):
        R = self.FW.transpose()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,3))

        S = np.zeros((self.meshO.nInternalCells, 3, 3))
        S[:, 0, 0] = self.X*self.X
        S[:, 0, 1] = self.X*self.Y
        S[:, 1, 1] = self.Y*self.Y
        S[:, 2, 2] = 1.
        res = evaluate(R.field, self.W, S, self)
        ref = np.zeros((self.meshO.nInternalCells, 3, 3))
        ref[:, 0, 0] = self.X*self.X
        ref[:, 1, 0] = self.X*self.Y
        ref[:, 1, 1] = self.Y*self.Y
        ref[:, 2, 2] = 1.
        checkArray(self, res, ref)

    def test_trace(self):
        R = self.FW.trace()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        S = np.zeros((self.meshO.nInternalCells, 3, 3))
        S[:, 0, 0] = self.X*self.X
        S[:, 0, 1] = self.X*self.Y
        S[:, 1, 1] = self.Y*self.Y
        S[:, 2, 2] = 1.
        res = evaluate(R.field, self.W, S, self)
        ref = (self.X*self.X + self.Y*self.Y + 1.).reshape(-1,1)
        checkArray(self, res, ref)

    def test_add(self):
        R = self.FU + self.FV
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        S = np.zeros((self.meshO.nInternalCells, 3))
        T = np.zeros((self.meshO.nInternalCells, 3))
        S[:,0], S[:,1] = self.X, self.Y
        T[:,0], T[:,1] = self.Y, -self.X
        res = evaluate(R.field, [self.U, self.V], [S, T], self)
        ref = np.zeros((self.meshO.nInternalCells, 3))
        ref[:,0] = self.X + self.Y
        ref[:,1] = self.Y - self.X
        checkArray(self, res, ref)

    def test_mul(self):
        R = self.FU * self.FV
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        S = np.zeros((self.meshO.nInternalCells, 3))
        T = np.zeros((self.meshO.nInternalCells, 3))
        S[:,0], S[:,1] = self.X, self.Y
        T[:,0], T[:,1] = self.Y, -self.X
        res = evaluate(R.field, [self.U, self.V], [S, T], self)
        ref = np.zeros((self.meshO.nInternalCells, 3))
        ref[:,0] = self.X * self.Y
        ref[:,1] = self.Y * -self.X
        checkArray(self, res, ref)

    def test_mul_vector(self):
        V = ad.bcmatrix()
        FV = Field('F', V, (1,))
        R = self.FU * FV
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        S = np.zeros((self.meshO.nInternalCells, 3))
        T = np.zeros((self.meshO.nInternalCells, 1))
        S[:,0], S[:,1] = self.X, self.Y
        T[:,0] = self.Y
        res = evaluate(R.field, [self.U, V], [S, T], self)
        ref = np.zeros((self.meshO.nInternalCells, 3))
        ref[:,0] = self.X*self.Y
        ref[:,1] = self.Y*self.Y
        checkArray(self, res, ref)

    def test_neg(self):
        R = self.FU.abs()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        T = np.zeros((self.meshO.nInternalCells, 3))
        T[:,0] = self.X
        T[:,1] = -1
        T[:,2] = 2.
        res = evaluate(R.field, self.U, T, self)
        ref = np.zeros_like(T)
        ref[:,0] = np.abs(self.X)
        ref[:,1] = 1.
        ref[:,2] = 2.
        checkArray(self, res, ref)

if __name__ == "__main__":
        unittest.main(verbosity=2, buffer=True)
