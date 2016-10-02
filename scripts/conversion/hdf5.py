#!/usr/bin/python2
from adFVM import parallel, config
import sys
import os
case = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == 'None':
    times = None
else:
    times = [float(x) for x in sys.argv[2:]]

config.hdf5 = False
from adFVM.field import IOField
from adFVM.mesh import Mesh

mesh = Mesh.create(case)
mesh.writeHDF5(case)
IOField.setMesh(mesh)

if not times:
    exit(0)

if len(times) == 0:
    times = mesh.getTimes()

for time in times:
    config.hdf5 = False
    fields = []
    with IOField.handle(time):
        for name in mesh.getFields(time):
            if name == 'polyMesh':
                # replace mesh boundary !!
                continue
            phi = IOField.readFoam(name)
            phi.partialComplete()
            fields.append(phi)

    config.hdf5 = True
    with IOField.handle(time, case=case):
        for phi in fields:
            phi.writeHDF5()
