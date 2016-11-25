import numpy as np

from adFVM import config
from adFVM.config import ad
from adFVM.density import RCF 

#primal = RCF('cases/forwardStep/', timeIntegrator='SSPRK', CFL=1.2, Cp=2.5, mu=lambda T: config.VSMALL*T)
primal = RCF('cases/forwardStep/', riemannSolver='eulerRoe', timeIntegrator='SSPRK', 
        CFL=0.6, Cp=2.5, mu=lambda T: 0.*T, faceReconstructor='AnkitENO')

def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    patchID = 'obstacle'
    startFace, endFace, cellStartFace, cellEndFace, _ = mesh.getPatchFaceCellRange(patchID)
    areas = mesh.areas[startFace:endFace]
    field = rhoE.field[cellStartFace:cellEndFace]
    return ad.sum(field*areas)

def perturb(fields, mesh, t):
    patchID = 'inlet'
    startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
    rho = np.zeros((mesh.nInternalCells, 1))
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoE = np.zeros((mesh.nInternalCells, 1))
    rhoU[mesh.owner[startFace:endFace], 0] += 0.1
    return rho, rhoU, rhoE

#nSteps = 4000
#writeInterval = 100
nSteps = 333
writeInterval = 333
startTime = 0.0
dt = 1e-4

