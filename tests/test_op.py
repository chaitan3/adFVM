
from op import grad, div, laplacian, snGrad

class TestOp(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.case = 'tests/convection/'
        self.mesh = Mesh.create(self.case)
        Field.setSolver(self)
        self.T = CellField('T', ad.zeros((self.mesh.nInternalCells, 1)))
        self.U = CellField('U', ad.zeros((self.mesh.nInternalCells, 3)))
        self.X = self.mesh.cellCentres[:, 0]
        self.Y = self.mesh.cellCentres[:, 1]
 
    def test_grad_scalar(self):
        self.T.field[:, 0] = self.X*self.Y + self.X**2 + self.Y**2 + self.X
        res = ad.value(grad(self.T).field)
        ref = np.zeros((self.mesh.nInternalCells, 3))
        x = self.X[:self.mesh.nInternalCells]
        y = self.Y[:self.mesh.nInternalCells]
        ref[:, 0] = y + 2*x + 1
        ref[:, 1] = x + 2*y
        checkSum(self, res, ref)

    def test_grad_vector(self):
        self.U.field[:, 0] = self.X*self.Y + self.X**2
        self.U.field[:, 1] = self.Y + self.Y**2 
        self.U.field[:, 2] = 1.
        res = ad.value(grad(self.U).field)
        ref = np.zeros((self.mesh.nInternalCells, 3, 3))
        x = self.X[:self.mesh.nInternalCells]
        y = self.Y[:self.mesh.nInternalCells]
        ref[:, 0, 0] = y + 2*x
        ref[:, 1, 0] = x
        ref[:, 1, 1] = 1 + 2*y
        checkSum(self, res, ref)

    def test_div(self):
        self.T.field[:, 0] = 1.
        self.U.field[:, 0] = self.X + np.sin(2*np.pi*self.X)*np.cos(2*np.pi*self.Y)
        self.U.field[:, 1] = self.Y**2 - np.cos(2*np.pi*self.X)*np.sin(2*np.pi*self.Y)
        self.U.field[:, 2] = self.X
        res = ad.value(div(self.T, self.U).field)
        y = self.Y[:self.mesh.nInternalCells]
        ref = (1 + 2*y).reshape(-1,1)
        checkSum(self, res, ref)

    def test_laplacian(self):
        self.T.field[:, 0] = self.X**2 + self.Y**2 + self.X*self.Y
        res = ad.value(laplacian(self.T, 1.).field)
        ref = 4.*np.ones((self.mesh.nInternalCells, 1))
        checkSum(self, res, ref, relThres=1e-2)


