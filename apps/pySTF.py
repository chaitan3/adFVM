#!/usr/bin/python2
from __future__ import print_function

from adFVM.op import div, laplacian
from adFVM.interp import central
#from adFVM.matop_petsc import div, ddt, laplacian, hybrid
from adFVM.solver import Solver

class STF(Solver):
    def __init__(self, case, **userConfig):
        super(STF, self).__init__(case, **userConfig)
        self.names = ['T', 'U']
        self.dimensions = [(1,), (3,)]
        
        self.DT = 0.01

    def equation(self, T, U):
        self.setBCFields([T, U])
        TF = central(T, self.mesh)
        UF = central(U, self.mesh)
        return [div(TF, UF) - laplacian(T, self.DT), 
                laplacian(U, 0)]

if __name__ == '__main__':
    solver = STF('cases/cylinder/')
    t = 2.0
    solver.readFields(t)
    solver.compile()
    solver.run(startTime=t, dt=0.001, nSteps=100)
