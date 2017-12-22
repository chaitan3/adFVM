import numpy as np

from adFVM import config
from adFVM.density import RCF 
from adFVM.mesh import Mesh
from adpy import tensor

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

    patch = mesh.boundary['cylinder']
    startFace, nFaces = patch['startFace'], patch['nFaces']
    meshArgs = _meshArgs(startFace)
    drag = tensor.Zeros((1,1))
    (drag,) = tensor.Kernel(objectiveDrag)(nFaces, (drag,))(U, T, p, *meshArgs, solver=solver)

    inputs = (drag,)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    (drag,) = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    return drag
   
primal = RCF('../cases/cylinder/',
#primal = RCF('cases/cylinder/',
             mu=lambda T: 2.5e-5,
             boundaryRiemannSolver='eulerLaxFriedrichs',
             objective = objective,
             fixedTimeStep = True,
)

def perturb(fields, mesh, t):
    #mid = np.array([-0.012, 0.0, 0.])
    #G = 100*np.exp(-3e4*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    mid = np.array([-0.001, 0.0, 0.])
    G = 1e3*np.exp(-1e5*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1, keepdims=1)**2)
    rho = G
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoU[:, 0] += G.flatten()*100
    rhoE = G*2e5
    return rho, rhoU, rhoE

parameters = 'source'

nSteps = 20
writeInterval = 10
startTime = 1.0
dt = 8e-9
