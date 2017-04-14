#!/usr/bin/python2
import sys
import glob
import h5py
import os

from adFVM import parallel, config
from adFVM.mesh import Mesh
from adFVM.field import IOField
config.hdf5 = True
case = sys.argv[1]

mesh = Mesh.create(case)
#mesh.writeFoam(case)

IOField.setMesh(mesh)
times = glob.glob(mesh.case + '*.hdf5')
#times = glob.glob(case + '*.hdf5')

# tested in serial
for timeF in times:
    if timeF.endswith('mesh.hdf5'): continue
    if os.path.exists(os.path.basename(timeF)[:-5]): continue
    fieldNames = h5py.File(timeF, 'r', driver='mpio', comm=parallel.mpi).keys()
    time = float(timeF[:-5].split('/')[-1])

    fields = []
    config.hdf5 = True
    with IOField.handle(time):
        for name in fieldNames:
            phi = IOField.readHDF5(name)
            phi.partialComplete()
            fields.append(phi)

    config.hdf5 = False
    with IOField.handle(time):
        for phi in fields:
            phi.writeFoam()
