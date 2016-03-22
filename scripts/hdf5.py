#!/usr/bin/python2

from mesh import Mesh
import parallel

mesh = Mesh()

import sys
mesh.caseDir = sys.argv[1]
mesh.case = mesh.caseDir + parallel.processorDirectory
mesh.readFoam('constant') 
mesh.writeHDF5()





