#!/usr/bin/python

from mesh import Mesh
from ops import interpolate, div
from field import write

import numpy as np
from matplotlib import pyplot as plt

case = 'test/'
mesh = Mesh(case)

field = np.zeros((mesh.nCells, 1))
mid = np.array([0.5, 0.5, 0.5])
for i in range(0, mesh.nCells):
    field[i] = np.exp(-10*np.linalg.norm(mid-mesh.cellCentres[i]))
    
U = 1.*np.ones((mesh.nFaces, 3))*np.array([1, 0, 0])
boundary = np.zeros((mesh.nFaces-mesh.nInternalFaces, 1))

t = 0
dt = 0.005
for i in range(0, 100):
    if i % 10 == 0:
        write(field, case + str(t) + '/', 'T')

    faceField = interpolate(field, mesh)
    divF = div(faceField, boundary, U, mesh)
    field += dt*-divF
    print field.max(), field.min()
    t += dt
    
