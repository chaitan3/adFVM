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
        phi.field[:mesh.nInternalCells] += 1e-6*np.random.randn(mesh.nInternalCells, phi.field.shape[1])*ref
        phi = IOField(name, phi.field, phi.dimensions, phi.boundary)
        phi.write()

