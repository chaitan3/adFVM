#!/usr/bin/python2
from __future__ import print_function
import unittest

from pyRCF import Solver

class TestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    def test_forwardStep(self):
        case = 'tests/forwardStep'
        solver = Solver(case, {'Cp': 2.5, 'mu': lambda T: T*0., 'CFL': 0.2})
        solver.run([0.0, 1e-3], 10000, 1000)

    def test_shockTube(self):
        case = 'tests/shockTube'
        solver = Solver(case, {'mu': lambda T: T*0., 'CFL': 0.6})
        solver.run([0.0, 1e-3], 100, 20)

    def test_convection(self):
        case = 'tests/convection'
        solver = Solver(case, {'mu': lambda T: 2.5e-5*T/T, 'CFL': 0.2})
        solver.run([1.0, 1e-3], 10000, 1000)

    def test_cylinder(self):
        case = 'tests/cylinder'
        solver = Solver(case)
        solver.run([0.0, 1e-3], 100000, 5000)



     
        
