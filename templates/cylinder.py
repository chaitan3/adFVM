from pyRCF import RCF 
from field import Field
from config import ad
from compat import norm
import numpy as np

primal = RCF('cases/cylinder/', mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)))
def objective(fields, mesh):
    rho, rhoU, rhoE = fields
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

def perturb():
    mesh = primal.mesh.origMesh
    mid = np.array([-0.0032, 0.0, 0.])
    G = 1e-4*np.exp(-1e2*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    rho = G
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoU[:, 0] += G.flatten()*100
    rhoE = G*3e5
    return rho, rhoU, rhoE

nSteps = 20000
writeInterval = 100
startTime = 2.0
dt = 1e-9


