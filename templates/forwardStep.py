import numpy as np

from adFVM import config
from adFVM.density import RCF 

def objectiveTest(U, T, p, *mesh, **options):
    mesh = Mesh.container(mesh)
    p0 = p.extract(mesh.neighbour)
    return (p0*mesh.areas).sum()

def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    patch = mesh.boundary['obstacle']
    startFace, nFaces = patch['startFace'], patch['nFaces']
    meshArgs = _meshArgs(startFace)
    test = tensor.Zeros((1,1))
    test = tensor.Kernel(objectiveTest)(nFaces, (test,))(U, T, p, *meshArgs)

    inputs = (test,)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    (test,) = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs
    return test

def perturb(fields, mesh, t):
    patchID = 'inlet'
    startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
    rho = np.zeros((mesh.nInternalCells, 1))
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoE = np.zeros((mesh.nInternalCells, 1))
    rhoU[mesh.owner[startFace:endFace], 0] += 0.1
    return rho, rhoU, rhoE

parameters = 'source'

nSteps = 100
writeInterval = 50
startTime = 0.0
dt = 1e-4

#adjParams = [1e-3, 'entropy', None]
