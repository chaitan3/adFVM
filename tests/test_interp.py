from interp import central, TVD_dual

class TestInterp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/convection/'
        self.mesh = Mesh.create(self.case)
        Field.setSolver(self)
        self.T = CellField('T', ad.zeros((self.mesh.nInternalCells, 1)))
        self.U = CellField('U', ad.zeros((self.mesh.nInternalCells, 3)))
        self.X = self.mesh.cellCentres[:, 0]
        self.XF = self.mesh.faceCentres[:, 0]
        self.Y = self.mesh.cellCentres[:, 1]
        self.YF = self.mesh.faceCentres[:, 1]
 

    def test_TVD_scalar(self):
        pass

    def test_TVD_vector(self):
        pass

    def test_interpolate(self):
        self.T.field[:, 0] = self.X + self.Y
        res = ad.value(interpolate(self.T).field)
        ref = (self.XF + self.YF).reshape(-1,1)
        check(self, res, ref)
     

