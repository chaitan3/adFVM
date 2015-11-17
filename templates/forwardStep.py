from pyRCF import RCF 
import config
from config import ad
import numpy as np
primal = RCF('cases/forwardStep/', timeIntegrator='SSPRK', CFL=1.2, Cp=2.5, mu=lambda T: config.VSMALL*T)

def objective(fields, mesh):
    rho, rhoU, rhoE = fields
    patchID = 'obstacle'
    startFace = mesh.boundary[patchID]['startFace']
    endFace = startFace + mesh.boundary[patchID]['nFaces']
    cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
    cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces
    areas = mesh.areas[startFace:endFace]
    field = rhoE.field[cellStartFace:cellEndFace]
    return ad.sum(field*areas)

def perturb(mesh):
    patchID = 'inlet'
    startFace = mesh.boundary[patchID]['startFace']
    endFace = startFace + mesh.boundary[patchID]['nFaces']
    cellStartFace = mesh.nInternalCells + startFace - mesh.nInternalFaces
    cellEndFace = mesh.nInternalCells + endFace - mesh.nInternalFaces
    stackedFields[cellStartFace:cellEndFace][:,1] += 0.1

nSteps = 20000
writeInterval = 500
startTime = 0.0
dt = 1e-4

