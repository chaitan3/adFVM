#!/usr/bin/python2
import sys
import numpy as np
import config

from adFVM import config
from adFVM.field import Field, IOField
from adFVM.mesh import Mesh

case, field, time1, time2 = sys.argv[1:]
time1 = float(time1)
time2 = float(time2)
mesh = Mesh.create(case)
Field.setMesh(mesh)

phi1 = IOField.read(field, mesh, time1)
phi2 = IOField.read(field, mesh, time2)

diff = abs(phi1.field-phi2.field)
phi1.info()
print field + ':', 'max diff:', diff.max(), 'relative:', diff.max()/phi1.field.max()
print 'close:', np.allclose(phi1.field, phi2.field)

#indices = (diff.max(axis=1) != 0.).astype(config.precision)
#indices = IOField('I', indices.reshape(-1,1), (1,))
#indices.partialComplete()
#indices.write(time1)
