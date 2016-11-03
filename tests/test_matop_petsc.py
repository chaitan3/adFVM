from adFVM.matop_petsc import laplacian, ddt
from adFVM.mesh import Mesh
from adFVM.interp import central
from adFVM.field import IOField
from adFVM.parallel import pprint

case = '../cases/cylinder/'
time = 3.0
field = 'mua'
case = '../cases/convection/'
time = 0.0
field = 'T'

DT = 0.01
dt = 0.001
nSteps = 100

mesh = Mesh.create(case)
IOField.setMesh(mesh)

with IOField.handle(time):
    T = IOField.read(field)
    T.partialComplete()
weight = central(T, mesh.origMesh)
weight.field[:] = DT
op = laplacian(T, weight)
op.eigenvalues()

for index in range(0, nSteps):
    pprint(index)
    T.old = T.field
    equation = ddt(T, dt) - laplacian(T, weight)
    T.field = equation.solve()
    T.partialComplete()
    time += dt

with IOField.handle(time):
    T.write()

