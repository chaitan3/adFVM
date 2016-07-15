import numpy as np

from adFVM import config
from adFVM.config import ad
from adFVM.compat import norm
from adFVM.density import RCF 

#primal = RCF('cases/convection/')
primal = RCF('cases/convection/', mu=lambda T: 0.*T)

def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    mid = np.array([0.75, 0.5, 0.5])
    G = ad.exp(-100*ad.sum((mid-mesh.cellCentres[:mesh.nInternalCells])**2, axis=1)).reshape((-1,1))*mesh.volumes[:mesh.nInternalCells]
    return ad.sum(rho.field[:mesh.nInternalCells]*G)

def source(fields, mesh, t):
    rho = np.zeros((mesh.nInternalCells, 1))
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoE = np.zeros((mesh.nInternalCells, 1))
    x = mesh.cellCentres[:mesh.nInternalCells, 0]
    y = mesh.cellCentres[:mesh.nInternalCells, 1]

    eps = 1e1
    G = eps*np.exp(-1e3*((x-0.5)**2+(y-0.5)**2))
    #G = eps*(1-np.sin(20*x))*(1-np.sin(20*y))
    rho[:,0] = 1.3*G
    rhoU[:,0] = 100*G
    rhoE[:,0] = 2e5*G
    return rho, rhoU, rhoE

nSteps = 20000
writeInterval = 500
#nSteps = 10
#writeInterval = 2
startTime = 0.0
dt = 1e-8
