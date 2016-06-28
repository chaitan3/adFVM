from pyRCF import RCF 
import config
from config import ad
from compat import norm
import numpy as np

primal = RCF('/home/talnikar/adFVM/cases/periodic_wake/')#, timeIntegrator='euler')

def objective(fields, mesh):
    res = fields[0].field.sum()
    return res

def source(fields, mesh, t):
    mesh = mesh.origMesh
    n = mesh.nInternalCells
    U = fields[1].field[:n]/fields[0].field[:n]
    x = mesh.cellCentres[:n, [0]]
    y = mesh.cellCentres[:n, [1]]
    rho = np.zeros((n, 1))
    rhoE = np.zeros((n, 1))

    rhoU = U*(1-np.sin(20*x))*(1-np.sin(20*y))

    return rho, rhoU, rhoE

#Steps = 20000
#riteInterval = 500
nSteps = 100000
writeInterval = 5000
startTime = 0.0
dt = 1e-8
