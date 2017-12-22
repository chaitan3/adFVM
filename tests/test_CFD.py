from __future__ import print_function

from adFVM import config
from adFVM.field import CellField, IOField
from adFVM.density import RCF
#from mms import source, solution

    # analytical inviscid comparison
def test_shockTube(self):
    case = '../cases/shockTube/'
    t = 0.0
    solver = RCF(case, mu=lambda T: T*0., faceReconstructor='AnkitENO', CFL=0.6)
    solver.readFields(t)
    solver.compile()
    timeRef = 0.006
    solver.run(startTime=t, endTime=timeRef, dt=1e-5)

    f = open(case + '/output')
    f.readline()
    f.readline()
    data = np.loadtxt(f)
    with IOField.handle(timeRef):
        rho = IOField.read('rho')
        p = IOField.read('p')
        U = IOField.read('U')
    #rho.complete()
    #p.complete()
    #U.complete()
    self.mesh = solver.mesh
    checkVolSum(self, rho.getInternalField(), data[:, 2].reshape((-1,1)), relThres=0.02)
    checkVolSum(self, p.getInternalField(), data[:, 3].reshape((-1,1)), relThres=0.02)
    checkVolSum(self, U.getInternalField()[:,0].reshape((-1,1)), data[:, 4].reshape((-1,1)),relThres=0.05)

if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
