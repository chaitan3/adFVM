import numpy as np
import sys, os

from adFVM import config
from adFVM.config import ad
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 

caseDir = '/projects/LESOpt/talnikar/vane-optim/'
nParam = 4

if not sys.argv[0].endswith('optim.py'):
    primal = RCF(CASEDIR, faceReconstructor='AnkitENO')

def dot(a, b):
    return ad.sum(a*b, axis=1, keepdims=True)

def spawnJob(args, parallel=True, cwd='.'):
    import subprocess
    #subprocess.call(args)
    if parallel:
        nProcs = 4096
        #nProcs = 16
        nProcsPerNode = 16
    else:
        nProcs = 1
        nProcsPerNode = 1
    #subprocess.check_call(['mpirun', '-np', nProcs] + args, cwd=cwd)
    with open('output.log', 'w') as f:
        subprocess.check_call(['runjob', 
                         '-n', str(nProcs), 
                         '-p', str(nProcsPerNode),
                         '--block', os.environ['COBALT_PARTNAME'],
                         #'--exp-env', 'BGLOCKLESSMPIO_F_TYPE', 
                         #'--exp-env', 'PYTHONPATH',
                         '--env_all',
                         '--verbose', 'INFO',
                         ':'] 
                        + args, stdout=f, stderr=f)

def genMeshParam(param, paramDir):
    foamDir = caseDir + '/foam/'
    sys.path.append(foamDir)
    from vane_profile import gen_mesh_param
    gen_mesh_param(param, foamDir, paramDir + '/foam/', spawnJob)
    import shutil
    shutil.move(paramDir + 'foam/mesh.hdf5', paramDir + 'mesh.hdf5')
    return

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
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    #print interCells.shape, interArea.sum()
    solver.postpro.extend([(ad.ivector(), interCells), (ad.bcmatrix(), interArea)])
    return solver.postpro[-2][0], solver.postpro[-1][0], normal
    
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
            perturbMesh.perturbation = mesh.getPerturbation(os.path.join(CASEDIR), 'grad{}'.format(index))
        return perturbMesh.perturbation
    return perturbMesh
perturb = []
for index in range(0, nParam):
    perturb.append(makePerturb(index))

nSteps = 100000
writeInterval = 5000
startTime = 1.0
dt = 1e-8

