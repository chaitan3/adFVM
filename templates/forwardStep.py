import numpy as np

from adFVM import config
from adFVM.config import ad
from adFVM.density import RCF 

#primal = RCF('cases/forwardStep/', timeIntegrator='SSPRK', CFL=1.2, Cp=2.5, mu=lambda T: config.VSMALL*T)
primal = RCF('cases/forwardStep/', riemannSolver='eulerRoe', timeIntegrator='SSPRK', CFL=1.2, Cp=2.5, mu=lambda T: 0.*T)

def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    patchID = 'obstacle'
    startFace, endFace, cellStartFace, cellEndFace, _ = mesh.getPatchFaceCellRange(patchID)
    areas = mesh.areas[startFace:endFace]
    field = rhoE.field[cellStartFace:cellEndFace]
    return ad.sum(field*areas)

def perturb(mesh):
    patchID = 'inlet'
    cellStartFace, cellEndFace, _ = mesh.getPatchCellRange(patchID)
    stackedFields[cellStartFace:cellEndFace][:,1] += 0.1

nSteps = 20000
writeInterval = 500
startTime = 0.0
dt = 1e-4

