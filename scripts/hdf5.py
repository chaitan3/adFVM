#!/usr/bin/python2
import parallel
import config
import sys
import os
case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]

config.hdf5 = False
from mesh import Mesh
mesh = Mesh.create(case)
mesh.writeHDF5(case)

config.hdf5 = True
from field import IOField
IOField.setMesh(mesh)
for time in times:
    fields = os.listdir(mesh.getTimeDir(time))
    IOField.openHandle(case, time)
    for name in fields:
        phi = IOField.readFoam(name, mesh, time)
        phi.partialComplete()
        phi.writeHDF5(case, time)
    IOField.closeHandle()
