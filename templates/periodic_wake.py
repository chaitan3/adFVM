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

    def blasius(x):
        a = 0.33
        m = 4.2
        c = a*x
        b = (c**m/(1+c**m))**(1/m)
        return b

    B = ((x-0.025)/0.025)**6
    up = y <= 0.005
    down = y >= 0.015
    w = 100*np.ones_like(y)
    w[up] = 100*blasius(6*(0.005-y[up])/0.005)
    w[down] = 100*blasius(6*(y[down]-0.015)/0.005)

    rhoU = B*(w-U)

    return rho, rhoU, rhoE

#Steps = 20000
#riteInterval = 500
nSteps = 100000
writeInterval = 5000
startTime = 0.0
dt = 1e-8
