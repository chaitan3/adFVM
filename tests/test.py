import unittest
import numpy as np

from adFVM import config
from adFVM.config import ad, T
config.unpickleFunction = False
config.pickleFunction = False
from adFVM.solver import SolverFunction

def evaluate(output, inputs, value, self):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(value, list):
        value = [value]
    f = SolverFunction(inputs, output, self, 'test', BCs=False, source=False)
    return f(*value)

def checkArray(self, res, ref, maxThres=1e-7, sumThres=1e-4):
    self.assertEqual(res.shape, ref.shape)
    diff = np.abs(res-ref)
    self.assertAlmostEqual(0, diff.max(), delta=maxThres)
    self.assertAlmostEqual(0, diff.sum(), delta=sumThres)

def checkVolSum(self, res, ref, relThres=1e-4):
    vols = self.mesh.origMesh.volumes
    if len(res.shape) == 3:
        vols = vols.flatten().reshape((-1,1,1))
    diff = np.abs(res-ref)*vols
    rel = diff.sum()/(ref*vols).sum()
    self.assertAlmostEqual(0, rel, delta=relThres)


