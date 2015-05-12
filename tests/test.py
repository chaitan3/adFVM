import sys
sys.path.append('../')
import unittest
import numpy as np
from config import ad, T

def evaluate(output, inputs, value):
    if not isinstance(inputs, list):
        inputs = [inputs]
    if not isinstance(value, list):
        value = [value]
    f = T.function(inputs, output)
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


