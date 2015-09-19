from pyRCF import RCF 
import config
from config import ad
from compat import norm
import numpy as np

#primal = RCF('.')
primal = RCF('/home/talnikar/foam/stator-rotor/laminar/')#, timeIntegrator='euler')
#primal = RCF('/master/home/talnikar/foam/stator-rotor/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/stator-rotor/les/')

# heat transfer
def objectiveHeatTransfer(fields, mesh):
    rho, rhoU, rhoE = fields
    solver = rhoE.solver
    U, T, p = solver.primitive(rho, rhoU, rhoE)

    res = 0
    for patchID in ['nozzle_suction', 'nozzle_pressure']:
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
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
    from compat import intersectPlane
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    print interCells.shape, interArea.sum()
    solver.postpro.extend([(ad.ivector(), interCells), (ad.bcmatrix(), interArea)])
    return solver.postpro[-2][0], solver.postpro[-1][0]
    
def objectivePressureLoss(fields, mesh):
    #if not hasattr(objectivePressureLoss, interArea):
    #    objectivePressureLoss.cells, objectivePressureLoss.area = getPlane(primal)
    #cells, area = objectivePressureLoss.cells, objectivePressureLoss.area
    ptin = 171371.
    cells, area = getPlane(primal)
    rho, rhoU, rhoE = fields
    solver = rhoE.solver
    g = solver.gamma
    U, T, p = solver.primitive(rho, rhoU, rhoE)
    c = (g*p/rho).sqrt() 
    Umag = U.mag()
    pi = p.field[cells]
    Mi = Umag.field[cells]/c.field[cells]
    pti = pi*(1 + 0.5*(g-1)*Mi*Mi)**(g/(g-1))
    res = ad.sum((ptin-pti)*area)
    return res 

objective = objectiveHeatTransfer
#objective = objectivePressureLoss

def perturb(mesh):
    mid = np.array([-0.08, 0.014, 0.005])
    G = 10*np.exp(-1e2*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
    #rho
    rho = G
    rhoU = np.zeros((mesh.nInternalCells, 3))
    rhoU[:, 0] = G.flatten()*100
    rhoE = G*2e5
    return rho, rhoU, rhoE

nSteps = 10
writeInterval = 2
startTime = 3.0
dt = 1e-8

