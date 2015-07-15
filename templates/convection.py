from pyRCF import RCF 
from config import ad
from compat import norm
import numpy as np

primal = RCF('cases/convection/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')

def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    mid = np.array([0.75, 0.5, 0.5])
    G = ad.exp(-100*ad.sum((mid-mesh.cellCentres[:mesh.nInternalCells])**2, axis=1)).reshape((-1,1))*mesh.volumes[:mesh.nInternalCells]
    return ad.sum(rho.field[:mesh.nInternalCells]*G)

def perturb(stackedFields, mesh, t):
    mid = np.array([0.5, 0.5, 0.5])
    if t == startTime:
        G = eps*ad.array(np.exp(-100*np.linalg.norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2).reshape(-1,1))
        rho.field[:mesh.nInternalCells] += G

nSteps = 1000
writeInterval = 100
startTime = 0.0
dt = 1e-8


