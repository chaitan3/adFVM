import numpy as np

from adFVM import config
from adFVM.config import ad
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 

#primal = RCF('cases/cylinder_steady/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*5e-5, (1,)))
#primal = RCF('cases/cylinder_per/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*5e-5, (1,)))
#primal = RCF('cases/cylinder_chaos_test/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)), boundaryRiemannSolver='eulerLaxFriedrichs')
primal = RCF('/home/talnikar/adFVM/cases/cylinder/chaotic/testing/', 
             timeIntegrator='SSPRK', 
             CFL=1.2, 
             mu=lambda T: T/T*2.5e-5,
             faceReconstructor='SecondOrder',
             boundaryRiemannSolver='eulerLaxFriedrichs'

)

def dot(a, b):
    return ad.sum(a*b, axis=1, keepdims=True)

# drag over cylinder surface
def objectiveDrag(fields, mesh):
    rho, rhoU, rhoE = fields
    patchID = 'cylinder'
    patch = mesh.boundary[patchID]
    start, end, nF = mesh.getPatchFaceRange(patchID)
    areas = mesh.areas[start:end]
    nx = ad.reshape(mesh.normals[start:end, 0], (-1, 1))
    cellStartFace = mesh.nInternalCells + start - mesh.nInternalFaces 
    cellEndFace = mesh.nInternalCells + end - mesh.nInternalFaces
    internalIndices = mesh.owner[start:end]
    start, end = cellStartFace, cellEndFace
    p = rhoE.field[start:end]*(primal.gamma-1)
    #deltas = (mesh.cellCentres[start:end]-mesh.cellCentres[internalIndices]).norm(2, axis=1, keepdims=True)
    dx = mesh.cellCentres[start:end]-ad.gather(mesh.cellCentres, internalIndices)
    deltas = ad.reshape(ad.sum(dx*dx, axis=1)**0.5, (nF,1))
    T = rhoE/(rho*primal.Cv)
    #mungUx = (rhoU.field[start:end, [0]]/rho.field[start:end]-rhoU.field[internalIndices, [0]]/rho.field[internalIndices])*primal.mu(T).field[start:end]/deltas
    mungUx = (ad.reshape(rhoU.field[start:end, 0], (nF,1))/rho.field[start:end]-ad.reshape(ad.gather(rhoU.field[:, 0], internalIndices), (nF,1))/ad.gather(rho.field, internalIndices))*primal.mu(T).field[start:end]/deltas
    return ad.sum((p*nx-mungUx)*areas)

def getPlane(solver):
    #point = np.array([0.0032,0.0,0.0], config.precision)
    point = np.array([0.032,0.0,0.0], config.precision)
    normal = np.array([1.,0.,0.], config.precision)
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    #print interCells.shape, interArea.sum()
    solver.postpro.extend([(ad.ivector(), interCells), (ad.bcmatrix(), interArea)])
    return solver.postpro[-2][0], solver.postpro[-1][0], normal
    
def objectivePressureLoss(fields, mesh):
    #if not hasattr(objectivePressureLoss, interArea):
    #    objectivePressureLoss.cells, objectivePressureLoss.area = getPlane(primal)
    #cells, area = objectivePressureLoss.cells, objectivePressureLoss.area
    ptin = 104190.
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

objective = objectiveDrag
#objective = objectivePressureLoss

#def makePerturb(scale):
#    def perturb(fields, mesh, t):
#        #mid = np.array([-0.012, 0.0, 0.])
#        #G = 100*np.exp(-3e4*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
#        mid = np.array([-0.01, 0.0, 0.])
#        G = scale*np.exp(-1e6*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
#        rho = G
#        rhoU = np.zeros((mesh.nInternalCells, 3))
#        rhoU[:, 0] += G.flatten()*100
#        rhoE = G*2e5
#        return rho, rhoU, rhoE
#    return perturb
# 
#perturb = [makePerturb(1e-5)]
#parameters = 'source'

def makePerturb(pt_per):
    def perturb(fields, mesh, t):
        return pt_per
    return perturb

#perturb = [makePerturb(0.1), makePerturb(0.2), makePerturb(0.4)]
perturb = [makePerturb(1.)]
parameters = ('BCs', 'p', 'left', 'U0')

nSteps = 50
writeInterval = 25
reportInterval = 1
startTime = 2.0
dt = 8e-8
