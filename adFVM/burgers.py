from adFVM.config import ad
from adFVM import interp
from adFVM.op import div, grad
from adFVM.interp import central
from adFVM.field import Field
from adFVM.solver import Solver

class Burgers(Solver):
    def __init__(self, case, **userConfig):
        super(Burgers, self).__init__(case, **userConfig)
        self.names = ['U']
        self.dimensions = [(1,)]
        self.faceReconstructor = interp.ENO(self)

    def getFlux(self, U, N):
        N = Field('nx', N.field[:,[0]], (1,))
        return 0.5*U*U*N

    def getRiemannFlux(self, UL, UR, N):
        UFL, UFR = self.getFlux(UL, N), self.getFlux(UR, N)
        a = (UFL-UFR)/(UL-UR)
        indices = (a.field < 0).nonzero()[0]
        UFL.setField(indices, UFR.field[indices])
        # entropy violating shock for rarefaction
        return UFL

    def equation(self, U):
        mesh = self.mesh
        self.setBCFields([U])
        #UF = central(U, self.mesh)
        #UFlux = self.getFlux(UF)
        gradU = grad(U, op=True, ghost=True)
        ULF, URF = self.faceReconstructor.dual(U, gradU)

        UFlux = Field('U', ad.zeros((mesh.nFaces,1)), U.dimensions)
        indices = self.faceReconstructor.indices
        NF = self.mesh.Normals.getField(indices)
        UIFlux = self.getRiemannFlux(ULF, URF, NF)
        UFlux.setField(indices, UIFlux)

        indices = self.faceReconstructor.Bindices
        cellIndices = indices - mesh.nInternalFaces + mesh.nInternalCells
        NF = self.mesh.Normals.getField(indices)
        UB = U.getField(cellIndices)
        UBFlux = self.getFlux(UB, NF)
        UFlux.setField(indices, UBFlux)
        self.local = UFlux.field
        self.remote = div(UFlux).field

        return [div(UFlux)]

    def setInitialCondition(self, U):
        mesh = self.mesh.origMesh
        n = mesh.nInternalCells
        #x = (mesh.cellCentres[:n,0]+5)/10
        #U.field[:n,0] = np.sin(x*2*np.pi)
        x = (mesh.cellCentres[:n,0]+5)/10
        U.field[:n/2,0] = 1
        U.field[n/2:n,0] = 0.5
        return

