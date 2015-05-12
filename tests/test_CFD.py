from __future__ import print_function
from test import *
import config

from field import CellField, IOField
from pyRCF import RCF
#from mms import source, solution

class TestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.mesh = None
    
    # analytical inviscid comparison
    def test_shockTube(self):
        skip
        config.fileFormat = 'ascii'
        case = '../cases/shockTube'
        solver = RCF(case, mu=lambda T: T*0., CFL=0.6)
        timeRef = 0.006
        solver.run(endTime=timeRef, dt=1e-5)

        f = open(case + '/output')
        f.readline()
        f.readline()
        data = np.loadtxt(f)
        rho = IOField.read('rho', solver.mesh, timeRef)    
        p = IOField.read('p', solver.mesh, timeRef)    
        U = IOField.read('U', solver.mesh, timeRef)    
        rho.complete()
        p.complete()
        U.complete()
        self.mesh = solver.mesh
        checkVolSum(self, rho.getInternalField(), data[:, 2].reshape((-1,1)), relThres=0.01)
        checkVolSum(self, p.getInternalField(), data[:, 3].reshape((-1,1)), relThres=0.01)
        checkVolSum(self, U.getInternalField()[:,0].reshape((-1,1)), data[:, 4].reshape((-1,1)),relThres=0.02)
        config.fileFormat = 'binary'

    # openfoam inviscid comparison
    def test_forwardStep(self):
        skip
        case = '../cases/forwardStep'
        solver = RCF(case, Cp=2.5, mu=lambda T: T*0., CFL=0.8)
        time = 1.7
        timeSteps, res = solver.run(endTime=time, writeInterval=500)
        timeRef = 10.0

        rho = IOField.read('rho', solver.mesh, time)    
        p = IOField.read('p', solver.mesh, time)    
        U = IOField.read('U', solver.mesh, time)    
        rho.complete()
        p.complete()
        U.complete()
        rhoRef = IOField.read('rho', solver.mesh, timeRef)    
        pRef = IOField.read('p', solver.mesh, timeRef)    
        URef = IOField.read('U', solver.mesh, timeRef)    
        rhoRef.complete()
        pRef.complete()
        URef.complete()
        self.mesh = solver.mesh
        checkVolSum(self, rho.getInternalField(), rhoRef.getInternalField(), relThres=0.01)
        checkVolSum(self, rhoU.getInternalField(), rhoURef.getInternalField(), relThres=0.01)
        checkVolSum(self, rhoE.getInternalField(), rhoERef.getInternalField(), relThres=0.01)

    # openfoam viscous comparison
    def test_cylinder(self):
        case = '../cases/cylinder'
        solver = RCF(case, mu=lambda T: 2.5e-5*T/T)
        time = 1 + 1e-4
        timeRef = 10
        solver.run(startTime=1.0, dt=5e-9, endTime=time, writeInterval=5000)
        rho = IOField.read('rho', solver.mesh, time)    
        p = IOField.read('p', solver.mesh, time)    
        U = IOField.read('U', solver.mesh, time)    
        rho.complete()
        p.complete()
        U.complete()
        rhoRef = IOField.read('rho', solver.mesh, timeRef)    
        pRef = IOField.read('p', solver.mesh, timeRef)    
        URef = IOField.read('U', solver.mesh, timeRef)    
        rhoRef.complete()
        pRef.complete()
        URef.complete()
        self.mesh = solver.mesh
        checkVolSum(self, rho.getInternalField(), rhoRef.getInternalField(), relThres=0.01)
        checkVolSum(self, rhoU.getInternalField(), rhoURef.getInternalField(), relThres=0.01)
        checkVolSum(self, rhoE.getInternalField(), rhoERef.getInternalField(), relThres=0.01)

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
        checkVolSum(self, rho.getInternalField(), rhoRef.getInternalField(), relThres=0.01)
        checkVolSum(self, rhoU.getInternalField(), rhoURef.getInternalField(), relThres=0.01)
        checkVolSum(self, rhoE.getInternalField(), rhoERef.getInternalField(), relThres=0.01)

if __name__ == "__main__":
        unittest.main(verbosity=2)
        #unittest.main(verbosity=2, buffer=True)
