from test import *

from adFVM.interp import central, FirstOrder, SecondOrder
from adFVM.op import grad

class TestInterp(TestAdFVM):
    def test_first_order(self):
        Field.setSolver(self)
        R = FirstOrder(self)
        R = R.dual(self.FU, None)
        ref = np.zeros((self.meshO.nFaces, 3))
        ref[:,0] = 10
        ref[:,1] = 4
        
        T = np.zeros((self.meshO.nCells, 3))
        T[:,0] = 10
        T[:,1] = 4
        res = evaluate([x.field for x in R], self.U, T, self)
        checkArray(self, res[0], ref)
        checkArray(self, res[1], ref)

    def test_second_order(self):
        Field.setSolver(self)
        R = SecondOrder(self)
        GR = grad(self.FU, op=True, ghost=True)
        R = R.dual(self.FU, GR)
        ref = np.zeros((self.meshO.nFaces, 3))
        ref[:,0] = np.sin(2*self.XF*np.pi)*np.sin(2*self.YF*np.pi)
        ref[:,1] = np.sin(2*self.YF*np.pi)
        
        T = np.zeros((self.meshO.nCells, 3))
        T[:,0] = np.sin(2*self.X*np.pi)*np.sin(2*self.Y*np.pi)
        T[:,1] = np.sin(2*self.Y*np.pi)
        res = evaluate([x.field for x in R], self.U, T, self)
        m = self.meshO.nInternalCells
        n = self.meshO.nInternalFaces
        res2 = evaluate(GR.field, self.U, T, self)
        checkArray(self, res[0], ref, maxThres=1e-3)
        checkArray(self, res[1], ref, maxThres=1e-3)

    def test_eno(self):
        from adFVM.burgers import Burgers
        config.compile_mode = T.compile.mode.Mode(linker='py', optimizer='None')

        solver = Burgers('../cases/burgers/')
        t = 0.0
        solver.readFields(t)
        solver.setInitialCondition(solver.fields[0])
        solver.compile()
        solver.writeFields(solver.fields, t)
        solver.run(startTime=t, dt=0.001, nSteps=1000, writeInterval=100)
        solver.readFields(1.)

        mesh = solver.mesh.origMesh
        res = solver.fields[0].field
        ref = np.ones_like(res)
        indices = (mesh.cellCentres[:mesh.nInternalCells,0] >= 0.75).nonzero()[0]
        ref[indices] = 0.5
        np.savetxt('res.txt', res)
        checkVolSum(self, res, ref, mesh=solver.mesh, relThres=1e-2)

    def test_ankit_eno(self):
        pass

    def test_central(self):
        Field.setSolver(self)
        R = central(self.FU, self.mesh)
        self.assertTrue(isinstance(R, Field))
        self.assertEqual(R.dimensions, (3,))

        T = np.zeros((self.meshO.nCells, 3))
        T[:,0] = self.X + self.Y
        T[:,1] = self.X * self.Y

        res = evaluate(R.field, self.U, T, self)
        ref = np.zeros((self.meshO.nFaces, 3))
        ref[:,0] = (self.XF + self.YF)
        ref[:,1] = (self.XF * self.YF)
        checkArray(self, res, ref)

if __name__ == "__main__":
        unittest.main(verbosity=2, buffer=True)
