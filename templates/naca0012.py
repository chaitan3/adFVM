import numpy as np

from adFVM import config
from adFVM.density import RCF 
from adFVM import tensor
from adFVM.mesh import Mesh

# drag over cylinder surface
def objectiveDrag(U, T, p, *mesh, **options):
    solver = options['solver']
    mesh = Mesh.container(mesh)
    U0 = U.extract(mesh.neighbour)[0]
    U0i = U.extract(mesh.owner)[0]
    p0 = p.extract(mesh.neighbour)
    T0 = T.extract(mesh.neighbour)
    nx = mesh.normals[0]
    mungUx = solver.mu(T0)*(U0-U0i)/mesh.deltas
    drag = (p0*nx-mungUx)*mesh.areas
    return drag.sum()

def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    patch = mesh.boundary['airfoil']
    startFace, nFaces = patch['startFace'], patch['nFaces']
    meshArgs = _meshArgs(startFace)
    drag = tensor.Zeros((1,1))
    (drag,) = tensor.Kernel(objectiveDrag)(nFaces, (drag,))(U, T, p, *meshArgs, solver=solver)

    inputs = (drag,)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    (drag,) = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    return drag

primal = RCF('/home/talnikar/adFVM/cases/naca0012/Re_12000/', 
#primal = RCF('/master/home/talnikar/adFVM-tf/cases/naca0012/adjoint_entropy/', 
             mu=lambda T: 3.4e-5,
             #mu=lambda T: 1e-9,
             #faceReconstructor='SecondOrder',
             #faceReconstructor='AnkitWENO',
             #boundaryRiemannSolver='eulerLaxFriedrichs',
             objective=objective,
             #fixedTimeStep=True
)

def makePerturb(pt_per):
    def perturb(fields, mesh, t):
        return pt_per
    return perturb

#perturb = [makePerturb(0.1), makePerturb(0.2), makePerturb(0.4)]
perturb = [makePerturb(0.4)]
parameters = ('BCs', 'p', 'inlet', 'pt')

nSteps = 200000
writeInterval = 50000
#reportInterval = 100
startTime = 1.0
startTime = 0.0
dt = 4e-10
