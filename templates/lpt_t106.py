import numpy as np

from adFVM import config
from adFVM.compat import intersectPlane
from adFVM.density import RCF 
from adFVM import tensor
from adFVM.mesh import Mesh



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

    patch = mesh.boundary['WALL']
    startFace, nFaces = patch['startFace'], patch['nFaces']
    meshArgs = _meshArgs(startFace)
    drag = tensor.Zeros((1,1))
    drag = tensor.Kernel(objectiveDrag)(nFaces, (drag,))(U, T, p, *meshArgs, solver=solver)

    inputs = (drag,)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    (drag,) = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    return drag

#primal = RCF('/home/talnikar/adFVM/cases/lpt_t106/', objective=objective, fixedTimeStep=True)
primal = RCF('/home/talnikar/adFVM/cases/lpt_t106/', objective=objective)

#def makePerturb(param, eps=1e-6):
#    def perturbMesh(fields, mesh, t):
#        if not hasattr(perturbMesh, 'perturbation'):
#            ## do the perturbation based on param and eps
#            #perturbMesh.perturbation = mesh.getPerturbation()
#            points = np.zeros_like(mesh.points)
#            #points[param] = eps
#            points[:] = eps*mesh.points
#            #points[:] = eps
#            perturbMesh.perturbation = mesh.getPointsPerturbation(points)
#        return perturbMesh.perturbation
#    return perturbMesh
#perturb = [makePerturb(1)]
##perturb = []
#
#parameters = 'mesh'

def makePerturb(mid):
    def perturb(fields, mesh, t):
        G = 1e0*np.exp(-1e2*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1, keepdims=1)**2)
        #rho
        rho = G
        rhoU = np.zeros((mesh.nInternalCells, 3), config.precision)
        rhoU[:, 0] = G.flatten()*100
        rhoE = G*2e5
        return rho, rhoU, rhoE
    return perturb
perturb = [makePerturb(np.array([-0.02, 0.01, 0.005], config.precision)),
           makePerturb(np.array([-0.08, -0.01, 0.005], config.precision))]

parameters = 'source'

#nSteps = 20000
#writeInterval = 50
#sampleInterval = 50
#reportInterval = 50
nSteps = 200000
writeInterval = 50000
#viscousInterval = 1
#sampleInterval = 10
reportInterval = 100
startTime = 0.0
dt = 1e-8
#runCheckpoints = 3

#adjParams = [1e-3, 'abarbanel', None]
#adjParams = [1e-3, 'entropy_jameson', None]
#adjParams = [1e-3, 'entropy_hughes', None]
#adjParams = [1e-3, 'uniform', None]
