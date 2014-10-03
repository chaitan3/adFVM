#!/usr/bin/python2
from __future__ import print_function
import unittest
import numpy as np
from utils import ad

from field import Field, CellField
from mesh import Mesh
from interp import interpolate, TVD_dual
from op import grad, div, laplacian, snGrad

class TestField(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/cylinder/'
        self.mesh = Mesh(self.case)
        self.p = CellField.read('p', self.mesh, 2.0)
        self.U = CellField.read('U', self.mesh, 2.0)
        self.gradU = grad(self.U, ghost=True)
        self.data = np.load(self.case + 'test_data.npz')

    def test_max(self):
        self.assertTrue(True)

    def test_component(self):
        res = ad.value(self.U.component(0).field)
        ref = self.data['component']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_magSqr(self):
        res = ad.value(self.U.magSqr().field)
        ref = self.data['magSqr']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_dot_vector(self):
        res = ad.value(interpolate(self.U).dotN().field)
        ref = self.data['dot_vector']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_outer(self):
        res = ad.value(interpolate(self.U).outer(self.mesh.Normals).field)
        ref = self.data['outer']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_dot_tensor(self):
        res = ad.value(self.gradU.dot(self.U).field)
        ref = self.data['dot_tensor']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_transpose(self):
        res = ad.value(self.gradU.transpose().field)
        ref = self.data['transpose']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_trace(self):
        res = ad.value(self.gradU.trace().field)
        ref = self.data['trace']
        self.assertAlmostEqual(0, np.abs(res-ref).max())
        
        
