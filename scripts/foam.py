#!/usr/bin/python2
import parallel
import config
import sys
import glob
import h5py
config.hdf5 = True
case = sys.argv[1]

from mesh import Mesh
mesh = Mesh.create(case)
#mesh.writeFoam(case)

from field import IOField
IOField.setMesh(mesh)
times = glob.glob(mesh.case + '*.hdf5')
for timeF in times:
    if timeF.endswith('mesh.hdf5'): continue
    fields = h5py.File(timeF, 'r', driver='mpio', comm=parallel.mpi).keys()
    time = float(timeF[:-5].split('/')[-1])
    IOField.openHandle(mesh.case, time)
    for name in fields:
        phi = IOField.readHDF5(name, mesh, time)
        phi.partialComplete()
        phi.writeFoam(mesh.case + parallel.processorDirectory, time)
    IOField.closeHandle()
