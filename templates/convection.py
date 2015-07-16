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

def perturb(mesh):
    eps = 1e-3
    mid = np.array([0.5, 0.5, 0.5])
    G = 1e-4*np.exp(-1e2*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    rho = G
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoE = G*0
    return rho, rhoU, rhoE

nSteps = 1000
writeInterval = 100
startTime = 0.0
dt = 1e-8
