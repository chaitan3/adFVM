#!/usr/bin/python
from adFVM import parallel, config
import sys
import os
case = sys.argv[1]
times = float(sys.argv[2])

config.hdf5 = False
from adFVM.field import IOField
from adFVM.mesh import Mesh

mesh = Mesh.create(case, currTime=times)
mesh.writeHDF5(case)
