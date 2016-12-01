from test import *

from adFVM.interp import central, FirstOrder, SecondOrder
from adFVM.op import grad

class TestTimeStep(unittest.TestCase):
    def test_rk(self):
        from adFVM.convection import LinearConvection
        #config.compile_mode = T.compile.mode.Mode(linker='py', optimizer='None')

        solver = LinearConvection('../cases/convection/')
        t = 0.0
        solver.readFields(t)
        T = solver.fields[0]
        mesh = solver.mesh.origMesh
        n = mesh.nInternalCells
        x = mesh.cellCentres[:n]
        T.field = np.sin(x[:,0]*2*np.pi).reshape(-1,1)
        ref = T.field.copy()
        solver.compile()
        solver.writeFields(solver.fields, t)
        solver.run(startTime=t, dt=0.001, nSteps=1000, writeInterval=100)
        solver.readFields(1.)

        mesh = solver.mesh.origMesh
        res = solver.fields[0].field

        checkVolSum(self, res, ref, mesh=solver.mesh, relThres=1e-2)

if __name__ == "__main__":
    unittest.main(verbosity=2, buffer=True)
