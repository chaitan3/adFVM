from pyRCF import RCF 
from config import ad
from compat import norm
import numpy as np

primal = RCF('cases/cylinder/', mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)))
def objective(fields):
    rho, rhoU, rhoE = fields
    mesh = rho.mesh
    patchID = 'cylinder'
    patch = mesh.boundary[patchID]
    nF = patch['nFaces']
    start, end = patch['startFace'], patch['startFace'] + nF
    areas = mesh.areas[start:end]
    nx = mesh.normals[start:end, 0].reshape((-1, 1))
    cellStartFace = mesh.nInternalCells + start - mesh.nInternalFaces
    cellEndFace = mesh.nInternalCells + end - mesh.nInternalFaces
    internalIndices = mesh.owner[start:end]
    start, end = cellStartFace, cellEndFace
    p = rhoE.field[start:end]*(primal.gamma-1)
    #deltas = (mesh.cellCentres[start:end]-mesh.cellCentres[internalIndices]).norm(2, axis=1, keepdims=True)
    deltas = (mesh.cellCentres[start:end]-mesh.cellCentres[internalIndices]).norm(2, axis=1).reshape((nF,1))
    T = rhoE/(rho*primal.Cv)
    #mungUx = (rhoU.field[start:end, [0]]/rho.field[start:end]-rhoU.field[internalIndices, [0]]/rho.field[internalIndices])*primal.mu(T).field[start:end]/deltas
    mungUx = (rhoU.field[start:end, 0].reshape((nF,1))/rho.field[start:end]-rhoU.field[internalIndices, 0].reshape((nF,1))/rho.field[internalIndices])*primal.mu(T).field[start:end]/deltas
    return ad.sum((p*nx-mungUx)*areas)/(nSteps + 1)

def perturb(stackedFields, t):
    mesh = primal.mesh.origMesh
    mid = np.array([-0.0032, 0.0, 0.])
    G = 1e-4*np.exp(-1e2*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    if t == startTime:
        stackedFields[:mesh.nInternalCells, 0] += G
        stackedFields[:mesh.nInternalCells, 1] += G*100
        stackedFields[:mesh.nInternalCells, 4] += G*2e5


