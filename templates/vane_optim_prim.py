import numpy as np

from adFVM import config
from adFVM.compat import intersectPlane
from adFVM.density import RCF 
from adFVM import tensor
from adFVM.mesh import Mesh

config.hdf5 = True
nParam = 4

# heat transfer
def objectiveHeatTransfer(U, T, p, weight, *mesh, **options):
    solver = options['solver']
    mesh = Mesh.container(mesh)
    Ti = T.extract(mesh.owner)
    Tw = 300.
    dtdn = (Tw-Ti)/mesh.deltas
    k = solver.Cp*solver.mu(Tw)/solver.Pr
    ht = k*dtdn*mesh.areas*weight
    w = mesh.areas*weight
    return ht.sum(), w.sum()

# pressure loss
@config.timeFunction('Time for finding intersection plane')
def getPlane(solver):
    point = np.array([0.052641,-0.1,0.005]).astype(config.precision)
    normal = np.array([1.,0.,0.]).astype(config.precision)
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    interCells = interCells.astype(np.int32)
    assert interCells.shape[0] == interArea.shape[0]
    nPlaneCells = interCells.shape[0]
    solver.extraArgs.append((tensor.IntegerScalar(), nPlaneCells))
    nPlaneCells = solver.extraArgs[-1][0]
    solver.extraArgs.append((tensor.Variable((nPlaneCells, 1), 'integer'), interCells))
    solver.extraArgs.append((tensor.Variable((nPlaneCells, 1)), interArea))
    return 

def objectivePressureLoss(U, T, p, cells, areas, **options):
    solver = options['solver']
    ptin = 175158.
    normal = np.array([1.,0.,0.])
    g = solver.gamma
    pi = p.extract(cells)
    Ti = T.extract(cells)
    Ui = U.extract(cells)
    rhoi = pi/(solver.Cv*Ti*(g- 1))
    ci = (g*pi/rhoi).sqrt()

    rhoUni = sum([rhoi*Ui[i]*normal[i] for i in range(0, 3)])
    Umagi = Ui.dot(Ui)
    Mi = Umagi.sqrt()/ci
    pti = pi*pow(1 + 0.5*(g-1)*Mi*Mi, g/(g-1))
    pl = (ptin-pti)*rhoUni*areas/ptin
    w = rhoUni*areas
    return pl.sum(), w.sum()

patches = ['pressure', 'suction']
def getWeights(solver):
    mesh = solver.mesh.symMesh
    for patchID in patches:
        patch = solver.mesh.boundary[patchID]
        #weights = np.zeros((patch['nFaces'], 1))
        centres = solver.mesh.faceCentres[patch['startFace']:patch['startFace'] + patch['nFaces']]
        if patchID == "pressure":
            weights = np.logical_and(centres[:,0] >= 0.033757, centres[:, 1] <= 0.04692)
        else:
            weights = np.logical_and(centres[:,0] >= 0.035241, centres[:, 1] <= 0.044337)
        nFaces = mesh.boundary[patchID]['nFaces']
        solver.extraArgs.append((tensor.Variable((nFaces, 1)), (weights*1.).astype(config.precision)))
    
def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    nPlaneCells, cells, areas = [x[0] for x in solver.extraArgs[:3]]
    pl, w = tensor.Zeros((1, 1)), tensor.Zeros((1, 1))
    pl, w = tensor.Kernel(objectivePressureLoss)(nPlaneCells, (pl, w))(U, T, p, cells, areas, solver=solver)

    _heatTransfer = tensor.Kernel(objectiveHeatTransfer)
    weights = [x[0] for x in solver.extraArgs[3:]]
    ht, w2 = tensor.Zeros((1, 1)), tensor.Zeros((1, 1))
    for index, patchID in enumerate(patches):
        patch = mesh.boundary[patchID]
        startFace, nFaces = patch['startFace'], patch['nFaces']
        meshArgs = _meshArgs(startFace)
        weight = weights[index]
        ht, w2 = _heatTransfer(nFaces, (ht, w2))(U, T, p, weight, *meshArgs, solver=solver)

    k = solver.mu(300)*solver.Cp/solver.Pr
    a = 0.4
    b = -0.71e-3/(120*k)/2000.

    # MPI ALLREDUCE
    #if not config.gpu:
    #    inputs = (pl, w, ht, w2)
    #    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    #    pl, w, ht, w2 = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs
    inputs = (pl, w, ht, w2)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    pl, w, ht, w2 = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    # then elemwise
    def _combine(pl, w, ht, w2):
        pl, w = pl.scalar(), w.scalar()
        ht, w2 = ht.scalar(), w2.scalar()
        obj = pl/w
        obj2 = ht/w2
        return a*obj + b*obj2
    return tensor.Kernel(_combine)(1)(pl, w, ht, w2)[0]

primal = RCF('./', objective=objective, fixedTimeStep=True)
#primal = RCF('/home/talnikar/adFVM/cases/vane/3d_10/', objective=objective)
#primal = RCF('/home/talnikar/adFVM/cases/vane/les/', objective=objective)
getPlane(primal)
getWeights(primal)

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
reportInterval = 1
nSteps = 300000
writeInterval = 100000
avgStart = 0
sampleInterval = 100
#nSteps = 10
#writeInterval = 5
#avgStart = 0
#sampleInterval = 1
startTime = 3.001
dt = 2e-8

# definition of 1 flow through time
# 4e-4s = (0.08m)/(200m/s)

