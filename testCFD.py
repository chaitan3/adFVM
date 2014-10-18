#!/usr/bin/python2
from __future__ import print_function
import unittest
import numpy as np

from field import CellField
from pyRCF import RCF

def checkSum(self, res, ref, vols, relThres=0.01):
    diff = np.abs(res-ref)*vols
    rel = diff.sum()/(ref*vols).sum()
    self.assertAlmostEqual(0, rel, delta=relThres)

class TestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass
    
    # analytical inviscid comparison
    def test_shockTube(self):
        skip
        case = 'tests/shockTube'
        solver = RCF(case, mu=lambda T: T*0., CFL=0.6)
        timeRef = 0.006
        solver.run(endTime=timeRef)

        f = open(case + '/output')
        f.readline()
        f.readline()
        data = np.loadtxt(f)
        rho = CellField.read('rho', timeRef)    
        p = CellField.read('p', timeRef)    
        U = CellField.read('U', timeRef)    
        vols = solver.mesh.volumes
        checkSum(self, rho.getInternalField(), data[:, 2].reshape((-1,1)), vols)
        checkSum(self, p.getInternalField(), data[:, 3].reshape((-1,1)), vols)
        checkSum(self, U.getInternalField()[:,0].reshape((-1,1)), data[:, 4].reshape((-1,1)), vols, relThres=0.02)

    # openfoam inviscid comparison
    def test_forwardStep(self):
        case = 'tests/forwardStep'
        solver = RCF(case, Cp=2.5, mu=lambda T: T*0., CFL=1.2)
        time = 1.7
        timeSteps, res = solver.run(endTime=time)
        timeRef = 10.0

        rho = CellField.read('rho', time)    
        p = CellField.read('p', time)    
        U = CellField.read('U', time)    
        rhoRef = CellField.read('rho', timeRef)    
        pRef = CellField.read('p', timeRef)    
        URef = CellField.read('U', timeRef)    
        vols = solver.mesh.volumes
        checkSum(self, rho.getInternalField(), rhoRef.getInternalField(), vols, relThres=0.05)
        checkSum(self, p.getInternalField(), pRef.getInternalField(), vols, relThres=0.07)
        checkSum(self, U.getInternalField(), URef.getInternalField(), vols, relThres=0.05)

    # analytical viscous comparison

    # openfoam viscous comparison
    def test_cylinder(self):
        skip
        case = 'tests/cylinder'
        solver = Solver(case)
        solver.run([0.0, 1e-3], 100000, 5000)

if __name__ == "__main__":
        unittest.main(verbosity=2)
        #unittest.main(verbosity=2, buffer=True)
