#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import time

from mesh import Mesh
from field import Field, CellField
#from op import div, ddt, laplacian
from op import div
from matop import ddt, laplacian, hybrid
from solver import implicit, forget

from utils import ad, pprint
from utils import Logger
import utils
logger = Logger(__name__)

case = 'tests/cyclic/'
mesh = Mesh(case)

t = 0.1
dt = 0.005
DT = 0.01

#initialize
T = CellField.zeros('T', mesh, (1,))
mid = np.array([0.5, 0.5, 0.5])
T.setInternalField(np.exp(-10*utils.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)).reshape(-1,1))
#T = Field.read('T', mesh, t)
U = 1.*ad.ones((mesh.nFaces, 3))*np.array([1., 0., 0])
U = Field('U', mesh, U)

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
    
    hybrid(equation, boundary, [T], dt)
    forget([T])
    t += dt
    t = round(t, 6)
    print()
    
