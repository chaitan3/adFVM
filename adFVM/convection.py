import numpy as np

from adFVM.config import ad
from adFVM.op import div, grad
from adFVM.interp import central
from adFVM.field import Field
from adFVM.solver import Solver

class LinearConvection(Solver):
    def __init__(self, case, **userConfig):
        super(LinearConvection, self).__init__(case, **userConfig)
        self.names = ['T']
        self.dimensions = [(1,)]

    def getFlux(self, T):
        mesh = self.mesh
        Uv = np.array([[1., 0, 0 ]])
        U = Field('U', Uv, (3,))
        #U = Field('U', ad.zeros((mesh.nFaces, 3)), (3,))
        #U.setField((0, mesh.nFaces), Uv)
        return (T*U).dotN()

    def equation(self, T):
        mesh = self.mesh
        TF = central(T, self.mesh)
        TFlux = self.getFlux(TF)
        return [div(TFlux)]


