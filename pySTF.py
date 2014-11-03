#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import time

from mesh import Mesh
from field import Field, CellField
#from op import div, ddt, laplacian
from matop import div, ddt, laplacian, hybrid
from solver import implicit, forget

from config import ad, Logger
from parallel import pprint
import config
logger = Logger(__name__)

case = 'tests/cyclic/'
mesh = Mesh(case)

t = 0.1
dt = 0.001
DT = 0.01

#initialize
Field.mesh = mesh
CellField.solver = mesh
T = CellField.zeros('T', (1,))
mid = np.array([0.5, 0.5, 0.5])
T.setInternalField(np.exp(-10*config.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)).reshape(-1,1))
#T = Field.read('T', mesh, t)
U = 1.*ad.ones((mesh.nFaces, 3))*np.array([1., 0., 0])
U = Field('U', U)

for i in range(0, 300):
    if i % 20 == 0:
        T.write(t)

    print('Simulation Time:', t, 'Time step:', dt)
    def equation(T):
        return [ddt(T, dt) + div(T, U) - laplacian(T, DT)]
    def boundary(TI):
        TN = CellField.copy(T)
        TN.setInternalField(TI)
        return [TN]
    
    T = hybrid(equation, boundary, [T], dt)[0]
    forget([T])
    t += dt
    t = round(t, 6)
    print()
    
