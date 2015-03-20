from __future__ import print_function
from test import *
import config

from field import Field, CellField
from mesh import Mesh

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
        self.assertTrue(False)

    def test_switch(self):
        self.assertTrue(False)

    def test_abs(self):
        self.assertTrue(False)

    def test_sign(self):
        self.assertTrue(False)

    def test_component(self):
        R = self.FU.component(0)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        T = np.random.rand(self.meshO.nFaces, 3)
        res = evaluate(R.field, self.U, T)
        ref = T[:,[0]]
        checkArray(self, res, ref)

    def test_magSqr(self):
        R = self.FU.magSqr()
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (1,))

        T = np.zeros((self.meshO.nInternalCells, 3))
        T[:, 0] = self.X
        T[:, 1] = self.Y
        res = evaluate(R.field, self.U, T)
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
        res = evaluate(R.field, [self.U, self.V], [S, T])
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
        res = evaluate(R.field, [self.U, self.V], [S, T])
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
        res = evaluate(R.field, [self.W, self.U], [S, T])
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
        res = evaluate(R.field, self.W, S)
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
        res = evaluate(R.field, self.W, S)
        ref = (self.X*self.X + self.Y*self.Y + 1.).reshape(-1,1)
        checkArray(self, res, ref)

    def test_add(self):
        self.assertTrue(False)

    def test_mul(self):
        self.assertTrue(False)

    def test_neg(self):
        self.assertTrue(False)

if __name__ == "__main__":
        unittest.main(verbosity=2, buffer=True)
