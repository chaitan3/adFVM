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
        self.case = 'tests/convection/'
        self.mesh = Mesh(self.case)
        Field.setSolver(self)
        self.Xf = self.mesh.cellCentres[:self.mesh.nInternalCells, 0]
        self.X = self.Xf.reshape(-1, 1)
        self.Yf = self.mesh.cellCentres[:self.mesh.nInternalCells, 1]
        self.Y = self.Yf.reshape(-1, 1)
        self.U = Field('U', ad.zeros((self.mesh.nInternalCells, 3)))
        self.V = Field('U', ad.zeros((self.mesh.nInternalCells, 3)))
        T = Field('T', ad.zeros((self.mesh.nInternalCells, 3, 3)))
        T.field[:, 0, 0] = self.Xf*self.Xf
        T.field[:, 0, 1] = self.Xf*self.Yf
        T.field[:, 1, 1] = self.Yf*self.Yf
        T.field[:, 2, 2] = 1.
        self.T = T

    def test_max(self):
        self.assertTrue(True)

    def test_component(self):
        self.U.field = self.X*np.array([[0.1,0.2,0.3]])
        res = ad.value(self.U.component(0).field)
        ref = 0.1*self.X
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_magSqr(self):
        self.U.field = self.X*np.array([[0.1,0.2,0.3]])
        res = ad.value(self.U.magSqr().field)
        ref = self.X**2*(0.1**2 + 0.2**2 + 0.3**2)
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_dot_vector(self):
        self.U.field[:, 0] = self.Xf
        self.U.field[:, 1] = self.Yf
        self.U.field[:, 2] = 0.5
        self.V.field[:, 0] = self.Yf
        self.V.field[:, 1] = -self.Xf
        self.V.field[:, 2] = 2.
        res = ad.value(self.U.dot(self.V).field)
        ref = 1.
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_outer(self):
        self.U.field[:, 0] = self.Xf
        self.U.field[:, 1] = self.Yf
        self.U.field[:, 2] = 0
        self.V.field[:, 0] = self.Yf
        self.V.field[:, 1] = -self.Xf
        self.V.field[:, 2] = 0
        res = ad.value(self.U.outer(self.V).field)
        ref = np.zeros((self.mesh.nInternalCells, 3, 3))
        ref[:, 0, 0] = self.Xf*self.Yf
        ref[:, 0, 1] = -self.Xf*self.Xf
        ref[:, 1, 0] = self.Yf*self.Yf
        ref[:, 1, 1] = -self.Xf*self.Yf
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_dot_tensor(self):
        self.U.field[:, 0] = 0.1
        self.U.field[:, 1] = 0.2
        self.U.field[:, 2] = 0.3
        res = ad.value(self.T.dot(self.U).field)
        ref = np.zeros((self.mesh.nInternalCells, 3))
        ref[:, 0] = self.T.field[:, 0, 0]*0.1 + self.T.field[:, 0, 1]*0.2
        ref[:, 1] = self.T.field[:, 1, 1]*0.2
        ref[:, 2] = self.T.field[:, 2, 2]*0.3
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_transpose(self):
        res = ad.value(self.T.transpose().field)
        ref = np.zeros((self.mesh.nInternalCells, 3, 3))
        ref[:, 0, 0] = self.Xf*self.Xf
        ref[:, 1, 0] = self.Xf*self.Yf
        ref[:, 1, 1] = self.Yf*self.Yf
        ref[:, 2, 2] = 1.
        self.assertAlmostEqual(0, np.abs(res-ref).max())

    def test_trace(self):
        res = ad.value(self.T.trace().field)
        ref = self.X*self.X + self.Y*self.Y + 1.
        self.assertAlmostEqual(0, np.abs(res-ref).max())
        
class TestInterp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/cylinder/'
        self.mesh = Mesh(self.case)
        Field.setSolver(self)
        self.rho = CellField.read('rho', 2.0)
        self.rhoU = CellField.read('rhoU', 2.0)
        self.rhoE = CellField.read('rhoE', 2.0)
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
        Field.setSolver(self)
        self.rho = CellField.read('rho', 2.0)
        self.rhoU = CellField.read('rhoU', 2.0)
        self.rhoE = CellField.read('rhoE', 2.0)
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






        
