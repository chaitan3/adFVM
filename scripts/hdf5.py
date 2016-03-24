#!/usr/bin/python2
import parallel
import config
import sys
import os
config.hdf5 = False
case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]

from mesh import Mesh
mesh = Mesh.create(case)
mesh.writeHDF5(case)

from field import IOField
IOField.setMesh(mesh)
for time in times:
    fields = os.listdir(mesh.getTimeDir(time))
    for name in fields:
        phi = IOField.readFoam(name, mesh, time)
        phi.partialComplete()
        phi.writeHDF5(case, time)
