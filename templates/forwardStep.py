import numpy as np

from adFVM import config
from adFVM.density import RCF 

primal = RCF('cases/forwardStep/', timeIntegrator='SSPRK', CFL=1.2, Cp=2.5, mu=lambda T: config.VSMALL*T)
#primal = RCF('/home/chaitukca/adFVM/cases/forwardStep/', riemannSolver='eulerRoe', timeIntegrator='SSPRK', 
#        CFL=1.2, Cp=2.5, mu=0., faceReconstructor='SecondOrder')

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

parameters = 'source'

#def perturb(fields, mesh, t):
#    if not hasattr(perturb, 'perturbation'):
#        ## do the perturbation based on param and eps
#        #perturbMesh.perturbation = mesh.getPerturbation()
#        points = np.zeros_like(mesh.parent.points)
#        points[0] = 1e-6
#        perturb.perturbation = mesh.parent.getPointsPerturbation(points)
#    return perturb.perturbation
#parameters = 'mesh'
#
#def perturb(fields, mesh, t):
#    return 1e-3
#
#parameters = ('BCs', 'U', 'inlet', 'value')

#nSteps = 4000
#writeInterval = 100
nSteps = 10
writeInterval = 2
startTime = 0.0
dt = 1e-4

adjParams = [1e-3, 'entropy', None]
