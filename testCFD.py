#!/usr/bin/python2
from __future__ import print_function
import unittest
import numpy as np

from field import CellField
from pyRCF import Solver

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
        solver = Solver(case, {'mu': lambda T: T*0., 'CFL': 0.6})
        timeSteps, res = solver.run([0.0, 1e-3], 250, 50)
        time = round(timeSteps[-1].sum(), 9)
        print(time)

        f = open(case + '/output')
        f.readline()
        f.readline()
        data = np.loadtxt(f)
        rho = CellField.read('rho', time)    
        p = CellField.read('p', time)    
        U = CellField.read('U', time)    
        vols = solver.mesh.volumes
        checkSum(self, rho.getInternalField(), data[:, 2].reshape((-1,1)), vols)
        checkSum(self, p.getInternalField(), data[:, 3].reshape((-1,1)), vols)
        checkSum(self, U.getInternalField()[:,0].reshape((-1,1)), data[:, 4].reshape((-1,1)), vols, relThres=0.02)

    # openfoam inviscid comparison
    def test_forwardStep(self):
        case = 'tests/forwardStep'
        solver = Solver(case, {'Cp': 2.5, 'mu': lambda T: T*0., 'CFL': 1.2})
        timeSteps, res = solver.run([0.0, 1e-3], 800, 100)
        time = round(timeSteps[-1].sum(), 9)
        timeRef = 10.0
        print(time)

        rho = CellField.read('rho', time)    
        p = CellField.read('p', time)    
        U = CellField.read('U', time)    
        rhoRef = CellField.read('rho', timeRef)    
        pRef = CellField.read('p', timeRef)    
        URef = CellField.read('U', timeRef)    
        vols = solver.mesh.volumes
        checkSum(self, rho.getInternalField(), rhoRef.getInternalField(), vols)
        checkSum(self, p.getInternalField(), pRef.getInternalField(), vols)
        checkSum(self, U.getInternalField(), URef.getInternalField(), vols)

    # analytical viscous comparison

    # openfoam viscous comparison
    def test_cylinder(self):
        exit2
        case = 'tests/cylinder'
        solver = Solver(case)
        solver.run([0.0, 1e-3], 100000, 5000)



     
        
