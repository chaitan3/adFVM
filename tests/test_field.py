from __future__ import print_function
import unittest
import numpy as np
from config import ad, T

from field import Field, CellField
from mesh import Mesh

from test import *

class TestField(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/convection/'
        self.mesh = Mesh.create(self.case)
        Field.setSolver(self)
        self.X = self.mesh.cellCentres[:self.mesh.nInternalCells, 0]
        self.Y = self.mesh.cellCentres[:self.mesh.nInternalCells, 1]
        self.U = Field('U', ad.zeros((self.mesh.nInternalCells, 3)))
        self.V = Field('V', ad.zeros((self.mesh.nInternalCells, 3)))
        self.W = Field('W', ad.zeros((self.mesh.nInternalCells, 3)))
        self.T = Field('T', ad.zeros((self.mesh.nInternalCells, 3, 3)))
        self.U.field[:, 0] = self.X
        self.U.field[:, 1] = self.Y
        self.U.field[:, 2] = 0.5
        self.V.field[:, 0] = self.Y
        self.V.field[:, 1] = -self.X
        self.V.field[:, 2] = 2.
        self.W.field[:, 0] = self.X*0.1
        self.W.field[:, 1] = self.X*0.2
        self.W.field[:, 2] = self.X*0.3
        self.T.field[:, 0, 0] = self.X*self.X
        self.T.field[:, 0, 1] = self.X*self.Y
        self.T.field[:, 1, 1] = self.Y*self.Y
        self.T.field[:, 2, 2] = 1.

    def test_max(self):
        self.assertTrue(True)

    def test_component(self):
        res = ad.value(self.W.component(0).field)
        ref = 0.1*self.X.reshape(-1, 1)
        check(self, res, ref)

    def test_magSqr(self):
        res = ad.value(self.W.magSqr().field)
        ref = self.X.reshape(-1,1)**2*(0.1**2 + 0.2**2 + 0.3**2)
        self.assertAlmostEqual(0, np.abs(res-ref).max())
        check(self, res, ref)

    def test_dot_vector(self):
        res = ad.value(self.U.dot(self.V).field)
        ref = 1.
        check(self, res, ref)

    def test_outer(self):
        res = ad.value(self.U.outer(self.V).field)
        ref = np.zeros((self.mesh.nInternalCells, 3, 3))
        ref[:, 0, 0] = self.X*self.Y
        ref[:, 0, 1] = -self.X*self.X
        ref[:, 0, 2] = 2*self.X
        ref[:, 1, 0] = self.Y*self.Y
        ref[:, 1, 1] = -self.X*self.Y
        ref[:, 1, 2] = 2*self.Y
        ref[:, 2, 0] = 0.5*self.Y
        ref[:, 2, 1] = -0.5*self.X
        ref[:, 2, 2] = 1.
        check(self, res, ref)

    def test_dot_tensor(self):
        res = ad.value(self.T.dot(self.W).field)
        ref = np.zeros((self.mesh.nInternalCells, 3))
        ref[:, 0] = (self.T.field[:, 0, 0]*0.1 + self.T.field[:, 0, 1]*0.2)*self.X
        ref[:, 1] = self.T.field[:, 1, 1]*0.2*self.X
        ref[:, 2] = self.T.field[:, 2, 2]*0.3*self.X
        check(self, res, ref)

    def test_transpose(self):
        res = ad.value(self.T.transpose().field)
        ref = np.zeros((self.mesh.nInternalCells, 3, 3))
        ref[:, 0, 0] = self.X*self.X
        ref[:, 1, 0] = self.X*self.Y
        ref[:, 1, 1] = self.Y*self.Y
        ref[:, 2, 2] = 1.
        check(self, res, ref)

    def test_trace(self):
        res = ad.value(self.T.trace().field)
        ref = (self.X*self.X + self.Y*self.Y + 1.).reshape(-1,1)
        check(self, res, ref)


if __name__ == "__main__":
        unittest.main(verbosity=2, buffer=True)
