#!/usr/bin/python2
import parallel
import config
import sys
config.hdf5 = False
case = sys.argv[1]

from mesh import Mesh
mesh = Mesh.create(case)
mesh.writeHDF5(case)

from field import IOField
IOField.setMesh(mesh)
time = 0.0
U = IOField.readFoam('U', mesh, time)
# testing hack
import numpy as np
U.field = np.zeros((mesh.origMesh.nCells, 3))
U.writeHDF5(case, time)
