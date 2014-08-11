#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import numpad as ad
import time

from mesh import Mesh
from field import Field
from ops import interpolate, div, ddt, solve, laplacian

case = 'test/'
mesh = Mesh(case)

#initialize
T = Field.zeros('T', mesh, mesh.nCells, 1)
mid = np.array([0.5, 0.5, 0.5])
T.setInternalField(np.exp(-10*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)).reshape(-1,1))
    
U = 1.*ad.ones((mesh.nFaces, 3))*np.array([1., 0., 0])
Uf = Field('U', mesh, U)

t = 0.1
dt = 0.005
DT = 0.01

#T.read(t)
for i in range(0, 300):
    print('Simulation Time:', t, 'Time step:', dt)
    if i % 20 == 0:
        T.write(t)
    T0 = Field.copy(T)
    eq = lambda T: ddt(T, T0, dt) + div(interpolate(T), Uf.field) - laplacian(T, DT)
    solve(eq, T)
    t += dt
    print()
    
