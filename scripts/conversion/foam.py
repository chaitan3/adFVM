#!/usr/bin/python2
import sys
import glob
import h5py

from adFVM import parallel, config
from adFVM.mesh import Mesh
from adFVM.field import IOField
config.hdf5 = True
case = sys.argv[1]

mesh = Mesh.create(case)
#mesh.writeFoam(case)

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
