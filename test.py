#!/usr/bin/python2
from __future__ import print_function
import unittest
import numpy as np
from config import ad

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
        
class TestInterp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/cylinder/'
        self.mesh = Mesh(self.case)
        self.rho = CellField.read('rho', self.mesh, 2.0)
        self.rhoU = CellField.read('rhoU', self.mesh, 2.0)
        self.rhoE = CellField.read('rhoE', self.mesh, 2.0)
        self.data = np.load(self.case + 'test_data.npz')
 
    def test_TVD_scalar(self):
        res = ad.value(TVD_dual(self.rho)[0].field)
        ref = self.data['TVD_scalar']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_TVD_vector(self):
        res = ad.value(TVD_dual(self.rhoU)[1].field)
        ref = self.data['TVD_vector']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_interpolate(self):
        res = ad.value(interpolate(self.rhoE).field)
        ref = self.data['interpolate']
        self.assertAlmostEqual(0, np.abs(res-ref).max())
     
class TestOp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/cylinder/'
        self.mesh = Mesh(self.case)
        self.rho = CellField.read('rho', self.mesh, 2.0)
        self.rhoU = CellField.read('rhoU', self.mesh, 2.0)
        self.rhoE = CellField.read('rhoE', self.mesh, 2.0)
        self.data = np.load(self.case + 'test_data.npz')
 
    def test_grad_scalar(self):
        res = ad.value(grad(self.rho).field)
        ref = self.data['grad_scalar']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_grad_vector(self):
        res = ad.value(grad(self.rhoU).field)
        ref = self.data['grad_vector']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_div(self):
        res = ad.value(div(self.rhoU).field)
        ref = self.data['div']
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_laplacian(self):
        res = ad.value(laplacian(self.rhoU/self.rho, 2.5e-5).field)
        ref = self.data['laplacian']
        self.assertAlmostEqual(0, np.abs(res-ref).max())






        
