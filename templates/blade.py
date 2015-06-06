from pyRCF import RCF 
import config
from config import ad
from compat import norm
import numpy as np

primal = RCF('.')
#primal = RCF('/home/talnikar/foam/blade/les/')
#primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')
def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    solver = rhoE.solver

    res = 0
    for patchID in ['suction', 'pressure']:
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
        cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
        cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces

        areas = mesh.areas[startFace:endFace]
        U, T, p = solver.primitive(rho, rhoU, rhoE)
        
        Ti = T.field[mesh.owner[startFace:endFace]] 
        Tw = 300*Ti/Ti
        deltas = (mesh.cellCentres[cellStartFace:cellEndFace]-mesh.cellCentres[patch.internalIndices]).norm(2, axis=1).reshape((nFaces, 1))
        dtdn = (Tw-Ti)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        dT = 120
        res += ad.sum(k*dtdn*areas)/(dT*ad.sum(areas)*(nSteps + 1) + config.VSMALL)
    return res

def perturb(stackedFields, mesh, t):
    mid = np.array([-0.08, 0.014, 0.005])
    G = 1e-3*np.exp(-1e5*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    if t == startTime:
        stackedFields[:mesh.nInternalCells, 0] += G
        stackedFields[:mesh.nInternalCells, 1] += G*100
        stackedFields[:mesh.nInternalCells, 4] += G*2e5

nSteps = 10
writeInterval = 2
startTime = 0.0
dt = 1e-9


