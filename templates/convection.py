from pyRCF import RCF 
from config import ad
from compat import norm
import numpy as np

#primal = RCF('cases/convection/')
primal = RCF('cases/cyclic/')

def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    mid = np.array([0.75, 0.5, 0.5])
    G = ad.exp(-100*ad.sum((mid-mesh.cellCentres[:mesh.nInternalCells])**2, axis=1)).reshape((-1,1))*mesh.volumes[:mesh.nInternalCells]
    return ad.sum(rho.field[:mesh.nInternalCells]*G)

def source(mesh):
    #eps = 1e-3
    #mid = np.array([0.5, 0.5, 0.5])
    #G = 1e-4*np.exp(-1e2*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    x = mesh.cellCentres[:mesh.nInternalCells, 0]
    y = mesh.cellCentres[:mesh.nInternalCells, 1]
    rho = np.zeros((mesh.nInternalCells, 1))
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoU[:,0] = 300.*(1-np.sin(20*x))*(1-np.sin(20*y))
    rhoE = np.zeros((mesh.nInternalCells, 1))
    return rho, rhoU, rhoE

nSteps = 20000
writeInterval = 500
startTime = 0.0
dt = 1e-8
