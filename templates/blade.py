from pyRCF import RCF 
from config import ad
import numpy as np

primal = RCF('/home/talnikar/foam/blade/les/')
#primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')
def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    solver = rhoE.solver

    res = 0
    for patchID in ['suction', 'pressure']:
        patch = rhoE.BC[patchID]
        start, end = patch.startFace, patch.endFace
        cellStart, cellEnd = patch.cellStartFace, patch.cellEndFace
        areas = mesh.areas[start:end]
        U, T, p = solver.primitive(rho, rhoU, rhoE)
        
        Ti = T.field[mesh.owner[start:end]] 
        Tw = 300*Ti/Ti
        deltas = (mesh.cellCentres[cellStart:cellEnd]-mesh.cellCentres[patch.internalIndices]).norm(2, axis=1).reshape((end-start, 1))
        dtdn = (Tw-Ti)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        dT = 120
        res += ad.sum(k*dtdn*areas)/(dT*ad.sum(areas)*(nSteps + 1) + config.VSMALL)
    return res

def perturb(stackedFields, mesh, t):
    mid = np.array([-0.08, 0.014, 0.005])
    G = 1e-3*np.exp(-1e5*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #G = 1e-4*np.exp(-1e2*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    if t == startTime:
        stackedFields[:mesh.nInternalCells, 0] += G
        stackedFields[:mesh.nInternalCells, 1] += G*100
        stackedFields[:mesh.nInternalCells, 4] += G*2e5

nSteps = 10
writeInterval = 2
startTime = 0.0
dt = 1e-9


