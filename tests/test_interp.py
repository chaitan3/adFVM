from __future__ import print_function
from test import *

from adFVM.field import Field, CellField
from adFVM.mesh import Mesh
from adFVM.interp import central, TVD

class TestInterp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = '../cases/convection/'
        self.mesh = Mesh.create(self.case)
        Field.setSolver(self)
        self.meshO = self.mesh.origMesh

        self.X = self.meshO.cellCentres[:, 0]
        self.Y = self.meshO.cellCentres[:, 1]
        self.XF = self.meshO.faceCentres[:, 0]
        self.YF = self.meshO.faceCentres[:, 1]

        self.U = ad.matrix()
        self.FU = CellField('F', self.U, (3,))

    def test_TVD_scalar(self):
        self.assertTrue(False)

    def test_TVD_vector(self):
        self.assertTrue(False)

    def test_central(self):
        R = central(self.FU, self.mesh)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        T = np.zeros((self.meshO.nCells, 3))
        T[:,0] = self.X + self.Y
        T[:,1] = self.X * self.Y

        res = evaluate(R.field, self.U, T, self)
        ref = np.zeros((self.meshO.nFaces, 3))
        ref[:,0] = (self.XF + self.YF)
        ref[:,1] = (self.XF * self.YF)
        checkArray(self, res, ref)

if __name__ == "__main__":
        unittest.main(verbosity=2, buffer=True)
