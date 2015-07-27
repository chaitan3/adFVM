from pyRCF import RCF 
import config
from config import ad
from compat import norm
import numpy as np

#primal = RCF('.')
primal = RCF('/home/talnikar/foam/blade/laminar-lowRe/')
#primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')
def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    solver = rhoE.solver
    U, T, p = solver.primitive(rho, rhoU, rhoE)

    res = 0
    for patchID in ['suction', 'pressure']:
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
        internalIndices = mesh.owner[startFace:endFace]

        areas = mesh.areas[startFace:endFace]
        deltas = mesh.deltas[startFace:endFace]

        Ti = T.field[internalIndices] 
        Tw = 300
        dtdn = (Tw-Ti)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        hf = ad.sum(k*dtdn*areas)
        #dT = 120
        #A = ad.sum(areas)
        #h = hf/((nSteps + 1)*dT*A)
        res += hf
    return res

def perturb(mesh):
    mid = np.array([-0.08, 0.014, 0.005])
    G = 10*np.exp(-1e2*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    rho = G
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoU[:, 0] = G.flatten()*100
    rhoE = G*2e5
    return rho, rhoU, rhoE

nSteps = 5000
writeInterval = 100
startTime = 3.0
dt = 1e-8


