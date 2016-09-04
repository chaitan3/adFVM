import unittest
import numpy as np

from adFVM import config
from adFVM.config import ad, T
config.unpickleFunction = False
config.pickleFunction = False
from adFVM.solver import SolverFunction
from adFVM.mesh import Mesh
from adFVM.field import Field, CellField

class TestAdFVM(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = '../cases/convection/'
        self.mesh = Mesh.create(self.case)
        Field.setSolver(self)
        self.meshO = self.mesh.origMesh
        self.postpro = []
        
        self.X = self.meshO.cellCentres[:, 0]
        self.Y = self.meshO.cellCentres[:, 1]
        self.XF = self.meshO.faceCentres[:, 0]
        self.YF = self.meshO.faceCentres[:, 1]

        self.U = ad.matrix()
        self.FU = CellField('F', self.U, (3,))
        self.V = ad.bcmatrix()
        self.FV = CellField('F', self.V, (1,))

def evaluate(output, inputs, value, self):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(value, list):
        value = [value]
    f = SolverFunction(inputs, output, self, 'test', BCs=False, source=False)
    return f(*value)

def checkArray(self, res, ref, maxThres=1e-7):#, sumThres=1e-4):
    self.assertEqual(res.shape, ref.shape)
    diff = np.abs(res-ref)
    self.assertAlmostEqual(0, diff.max(), delta=maxThres)
    #self.assertAlmostEqual(0, diff.sum(), delta=sumThres)

def checkVolSum(self, res, ref, relThres=1e-4, mesh=None):
    if not mesh:
        mesh = self.mesh
    vols = mesh.origMesh.volumes
    if len(res.shape) == 3:
        vols = vols.flatten().reshape((-1,1,1))
    diffV = np.abs(res-ref)*vols
    refV = np.abs(ref)*vols
    rel = diffV.sum()/refV.sum()
    self.assertAlmostEqual(0, rel, delta=relThres)


