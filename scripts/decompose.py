#!/usr/bin/python
import sys, os

from adFVM.mesh import Mesh
from adFVM.field import IOField
case = sys.argv[1]
nprocs = int(sys.argv[2])
times = [float(x) for x in sys.argv[3:]]

mesh = Mesh.create(case)
data = mesh.decompose(nprocs)

IOField.setMesh(mesh)
for time in times:
    fields = mesh.getFields(time)
    with IOField.handle(time):
        for name in fields:
            phi = IOField.readFoam(name)
            phi.partialComplete()
            phi.decompose(time, data)
