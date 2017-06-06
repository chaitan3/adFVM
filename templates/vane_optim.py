import numpy as np
import sys, os

from adFVM import config
from adFVM.config import ad
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 

config.hdf5 = True
caseDir = './'
primal = RCF(caseDir, objective='pressureLoss', objectivePLInfo={})
nParam = 4

def dot(a, b):
    return ad.sum(a*b, axis=1, keepdims=True)

# heat transfer
def objectiveHeatTransfer(fields, mesh):
    rho, rhoU, rhoE = fields
    solver = rhoE.solver
    U, T, p = solver.primitive(rho, rhoU, rhoE)

    res = 0
    for patchID in ['suction', 'pressure']:
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
        internalIndices = mesh.owner[startFace:endFace]

        areas = mesh.areas[startFace:endFace]
        deltas = mesh.deltas[startFace:endFace]

        Ti = T.field[internalIndices] 
        Tw = 300
        dtdn = (Tw-Ti)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        hf = ad.sum(k*dtdn*areas)
        #dT = 120
        #A = ad.sum(areas)
        #h = hf/((nSteps + 1)*dT*A)
        res += hf
    return res

# pressure loss
def getPlane(solver):
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    ptin = 175158.
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    return {'cells':interCells.astype(np.int32), 
            'areas': interArea, 
            'normal': normal, 
            'ptin': ptin
           }
primal.defaultConfig["objectivePLInfo"] = getPlane(primal)
    
def objectivePressureLoss(fields, mesh):
    #if not hasattr(objectivePressureLoss, interArea):
    #    objectivePressureLoss.cells, objectivePressureLoss.area = getPlane(primal)
    #cells, area = objectivePressureLoss.cells, objectivePressureLoss.area
    ptin = 175158.
    #actual ptin = 189718.8
    cells, area, normal = getPlane(primal)
    rho, rhoU, rhoE = fields
    solver = rhoE.solver
    g = solver.gamma
    U, T, p = solver.primitive(rho, rhoU, rhoE)
    pi, rhoi, Ui = p.field[cells], rho.field[cells], U.field[cells]
    rhoUi, ci = rhoi*Ui, ad.sqrt(g*pi/rhoi)
    rhoUni, Umagi = dot(rhoUi, normal), ad.sqrt(dot(Ui, Ui))
    Mi = Umagi/ci
    pti = pi*(1 + 0.5*(g-1)*Mi*Mi)**(g/(g-1))
    #res = ad.sum((ptin-pti)*rhoUni*area)/(ad.sum(rhoUni*area) + config.VSMALL)
    res = ad.sum((ptin-pti)*rhoUni*area)#/(ad.sum(rhoUni*area) + config.VSMALL)
    return res 

#objective = objectiveHeatTransfer
objective = objectivePressureLoss

def makePerturb(index):
    def perturbMesh(fields, mesh, t):
        if not hasattr(perturbMesh, 'perturbation'):
            perturbMesh.perturbation = mesh.getPerturbation(caseDir + 'grad{}/'.format(index))
        return perturbMesh.perturbation
    return perturbMesh
perturb = []
for index in range(0, nParam):
    perturb.append(makePerturb(index))

parameters = 'mesh'
nSteps = 10
writeInterval = 5
reportInterval = 1
avgStart = 2
startTime = 3.0
dt = 1e-8

