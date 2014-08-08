#!/usr/bin/python

from mesh import Mesh
from field import Field, FaceField
from ops import interpolate, div

import numpy as np
from matplotlib import pyplot as plt

case = 'test/'
mesh = Mesh(case)

T = Field.zeros('T', mesh, 1)
mid = np.array([0.5, 0.5, 0.5])
for i in range(0, mesh.nInternalCells):
    T.field[i] = np.exp(-10*np.linalg.norm(mid-mesh.cellCentres[i]))
    
U = 1.*np.ones((mesh.nFaces, 3))*np.array([0.7, 0.3, 0])
Uf = FaceField('U', mesh, U)

t = 0.1
dt = 0.005
for i in range(0, 100):
    if i % 10 == 0:
        T.write(t)

    T.field[:mesh.nInternalCells] += dt*-div(interpolate(T), Uf)
    # update ghost cells
    print T.field.max(), T.field.min()
    t += dt
    
