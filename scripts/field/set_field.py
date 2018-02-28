#!/usr/bin/python2
import sys, os
import numpy as np

from adFVM import config
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh

case = sys.argv[1]
time1, time2 = sys.argv[2:4]
mesh = Mesh.create(case)
Field.setMesh(mesh)

fields = ['p', 'T', 'U']
fieldsRef = [2e5, 300, 100]
for name, ref in zip(fields, fieldsRef):
    with IOField.handle(float(time1)):
        phi = IOField.read(name)
        phi.partialComplete()
    with IOField.handle(float(time2)):
        mid = np.array([-0.02, 0.01, 0.005])
        G = 1e0*np.exp(-3e3*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1, keepdims=1)**2)
        phi.field[:mesh.nInternalCells] = G*ref

        phi = IOField(name, phi.field, phi.dimensions, phi.boundary)
        phi.write()

