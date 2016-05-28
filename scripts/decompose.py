from mesh import Mesh
from field import IOField

import sys

case = sys.argv[1]
nprocs = int(sys.argv[2])
times = [float(x) for x in sys.argv[3:]]

mesh = Mesh.create(case)
mesh.decompose(nprocs)

IOField.setMesh(mesh)
for time in times:
    fields = os.listdir(mesh.getTimeDir(time))
    IOField.openHandle(case, time)
    for name in fields:
        phi = IOField.readFoam(name, mesh, time)
        phi.partialComplete()
        phi.decompose(case, time)
    IOField.closeHandle()
