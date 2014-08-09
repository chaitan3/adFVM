#!/usr/bin/python

from mesh import Mesh
from field import Field
from ops import interpolate, div, ddt, solve

import numpy as np
import numpad as ad

case = 'test/'
mesh = Mesh(case)

T = Field.zeros('T', mesh, mesh.nCells, 1)
mid = np.array([0.5, 0.5, 0.5])
for i in range(0, mesh.nInternalCells):
    T.field[i] = np.exp(-10*np.linalg.norm(mid-mesh.cellCentres[i]))
    
U = 1.*ad.ones((mesh.nFaces, 3))*np.array([0.5, .5, 0])
Uf = Field('U', mesh, U)

t = 0.1
dt = 0.005
for i in range(0, 1):
    print t
    if i % 20 == 0:
        T.write(t)
    eq = lambda T: ddt(T, dt) + div(interpolate(T), Uf)
    solve(eq, T)
    t += dt
    
