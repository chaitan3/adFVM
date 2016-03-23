#!/usr/bin/python2
from mesh import Mesh
import parallel
import sys

mesh = Mesh()

mesh.caseDir = sys.argv[1]
mesh.case = mesh.caseDir + parallel.processorDirectory
mesh.readFoam('constant') 
mesh.writeHDF5()
