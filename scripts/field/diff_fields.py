#!/usr/bin/python2
import sys
import numpy as np

from adFVM import config, parallel
from adFVM.parallel import pprint
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh

case1, case2, field = sys.argv[1:]
mesh1 = Mesh.create(case1)
mesh2 = Mesh.create(case2)

times1 = mesh1.getTimes()
times2 = mesh2.getTimes()

for time1, time2 in zip(times1, times2):
    Field.setMesh(mesh1)
    with IOField.handle(time1):
        phi1 = IOField.read(field)
    Field.setMesh(mesh2)
    with IOField.handle(time2):
        phi2 = IOField.read(field)
    diff = np.abs(phi1.field-phi2.field)
    norm = np.sqrt(parallel.sum(diff**2*mesh1.origMesh.volumes))
    pprint(parallel.min(diff))
    pprint('norm:', norm)

exit()

case, field, time1, time2 = sys.argv[1:]
time1 = float(time1)
time2 = float(time2)
mesh = Mesh.create(case)
Field.setMesh(mesh)

phi1 = IOField.read(field, mesh, time1)
phi2 = IOField.read(field, mesh, time2)

diff = abs(phi1.field-phi2.field)
ref = phi1.field.max()
phi1.info()
print field + ':', 'max diff:', diff.max(), 'relative:', diff.max()/ref
print 'close:', np.allclose(phi1.field, phi2.field)
