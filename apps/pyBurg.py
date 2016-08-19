#!/usr/bin/python2
from __future__ import print_function
import numpy as np

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

    def getFlux(self, U):
        return 0.5*U*U

    def getRiemannFlux(self, UL, UR):
        UFL, UFR = self.getFlux(UL), self.getFlux(UR)
        a = (UFL-UFR)/(UL-UR)
        UF = UFL
        indices = (a.field > 0).nonzero()[0]
        UFL.setField(indices, UFR.field[indices])
        return UFL

    def equation(self, U):
        mesh = self.mesh
        self.setBCFields([U])
        #UF = central(U, self.mesh)
        #UFlux = self.getFlux(UF)
        gradU = grad(U, op=True, ghost=True)
        ULF, URF = self.faceReconstructor.dual(U, gradU)

        UFlux = Field('U', ad.zeros((mesh.nFaces,1)), U.dimensions)
        UIFlux = self.getRiemannFlux(ULF, URF)
        indices = self.faceReconstructor.indices
        UFlux.setField(indices, UIFlux)

        indices = self.faceReconstructor.Bindices
        UB = U.getField(indices)
        UBFlux = self.getFlux(UB)
        UFlux.setField(indices, UBFlux)

        return [div(UFlux)]

    def setInitialCondition(self, U):
        mesh = self.mesh.origMesh
        n = mesh.nInternalCells
        x = (mesh.cellCentres[:n,0]+5)/10
        U.field[:n,0] = np.sin(x*2*np.pi)
        return

if __name__ == '__main__':
    solver = Burgers('cases/burgers/')
    t = 0.0
    solver.readFields(t)
    solver.setInitialCondition(solver.fields[0])
    solver.writeFields(solver.fields, t)
    solver.compile()
    solver.run(startTime=t, dt=0.001, nSteps=1000, writeInterval=100)
