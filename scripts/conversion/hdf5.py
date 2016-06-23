#!/usr/bin/python2
import parallel
import config
import sys
import os
case = sys.argv[1]
times = [float(x) for x in sys.argv[2:]]

from field import IOField
from mesh import Mesh

config.hdf5 = False
mesh = Mesh.create(case)
mesh.writeHDF5(case)
IOField.setMesh(mesh)

if len(times) == 0:
    times = mesh.getTimes()

for time in times:
    config.hdf5 = False
    fields = []
    IOField.openHandle(time)
    for name in os.listdir(mesh.getTimeDir(time)):
        phi = IOField.readFoam(name)
        phi.partialComplete()
        fields.append(phi)
    IOField.closeHandle()

    config.hdf5 = True
    IOField.openHandle(time, case=case)
    for phi in fields:
        phi.writeHDF5()
    IOField.closeHandle()
