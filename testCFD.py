#!/usr/bin/python2
from __future__ import print_function
import unittest
import numpy as np

from field import CellField
from pyRCF import RCF
from mms import source, solution

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
        skip
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
    def test_mms(self):
        case = 'tests/convection'
        timeRef = 0.1
        solver = RCF(case, mu=lambda T: 10.*T/T, Pr=1.0, source=source, timeIntegrator='implicit', stepFactor=1.0001)
        solver.run(endTime=timeRef, dt=1e-5, writeInterval=100)
        rhoRef, rhoURef, rhoERef = solution(timeRef, solver.mesh)

        rho = CellField.read('rho', timeRef)    
        rhoU = CellField.read('rhoU', timeRef)    
        rhoE = CellField.read('rhoE', timeRef)    
        vols = solver.mesh.volumes
        checkSum(self, rho.getInternalField(), rhoRef.getInternalField(), vols, relThres=0.01)
        checkSum(self, rhoU.getInternalField(), rhoURef.getInternalField(), vols, relThres=0.01)
        checkSum(self, rhoE.getInternalField(), rhoERef.getInternalField(), vols, relThres=0.01)

    # openfoam viscous comparison
    def test_cylinder(self):
        skip
        case = 'tests/cylinder'
        solver = RCF(case, mu=lambda T: 2.5e-5*T/T)
        time = 1 + 1e-4
        timeRef = 10.0
        #solver.run(startTime=1.0, endTime=time, writeInterval=1000)
        rho = CellField.read('rho', time)    
        p = CellField.read('p', time)    
        U = CellField.read('U', time)    
        rhoRef = CellField.read('rho', timeRef)    
        pRef = CellField.read('p', timeRef)    
        URef = CellField.read('U', timeRef)    
        vols = solver.mesh.volumes
        checkSum(self, rho.getInternalField(), rhoRef.getInternalField(), vols, relThres=0.01)
        checkSum(self, p.getInternalField(), pRef.getInternalField(), vols, relThres=0.01)
        checkSum(self, U.getInternalField(), URef.getInternalField(), vols, relThres=0.02)



if __name__ == "__main__":
        unittest.main(verbosity=2)
        #unittest.main(verbosity=2, buffer=True)
