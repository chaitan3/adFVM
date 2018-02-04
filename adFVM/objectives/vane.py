import numpy as np
from adFVM import config
from adFVM.compat import intersectPlane
from adpy import tensor
from adFVM.mesh import Mesh

def objectiveHeatTransferWeighting(weight, *mesh):
    mesh = Mesh.container(mesh)
    return (mesh.areas*weight).sum()

def objectiveHeatTransfer(U, T, p, weight, w, *mesh, **options):
    w = w.scalar()
    solver = options['solver']
    mesh = Mesh.container(mesh)
    Ti = T.extract(mesh.owner)
    Tw = 300.
    dtdn = (Tw-Ti)/mesh.deltas
    k = solver.Cp*solver.mu(Tw)/solver.Pr
    ht = k*dtdn*mesh.areas*weight
    return (ht/w).sum()

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
    solver.extraArgs.append((tensor.StaticIntegerVariable((nPlaneCells, 1)), interCells))
    solver.extraArgs.append((tensor.StaticVariable((nPlaneCells, 1)), interArea))
    return 

def objectivePressureLossWeighting(U, T, p, cells, areas, **options):
    solver = options['solver']
    normal = np.array([1.,0.,0.])
    g = solver.gamma
    pi = p.extract(cells)
    Ti = T.extract(cells)
    Ui = U.extract(cells)
    rhoi = pi/(solver.Cv*Ti*(g- 1))

    rhoUni = sum([rhoi*Ui[i]*normal[i] for i in range(0, 3)])
    w = rhoUni*areas
    return w.sum()

def objectivePressureLoss(U, T, p, cells, areas, w, **options):
    w = w.scalar()
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
    return (pl/w).sum()



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
        solver.extraArgs.append((tensor.StaticVariable((nFaces, 1)), (weights*1.).astype(config.precision)))
    
def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    nPlaneCells, cells, areas = [x[0] for x in solver.extraArgs[:3]]
    w = tensor.Zeros((1, 1))
    w = tensor.Kernel(objectivePressureLossWeighting)(nPlaneCells, (w,))(U, T, p, cells, areas, solver=solver)

    weights = [x[0] for x in solver.extraArgs[3:]]
    w2 = tensor.Zeros((1, 1))
    _heatTransferWeighting = tensor.Kernel(objectiveHeatTransferWeighting)
    for index, patchID in enumerate(patches):
        patch = mesh.boundary[patchID]
        startFace, nFaces = patch['startFace'], patch['nFaces']
        meshArgs = _meshArgs(startFace)
        weight = weights[index]
        w2 = _heatTransferWeighting(nFaces, (w2,))(weight, *meshArgs)

    inputs = (w, w2)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    w, w2 = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    pl = tensor.Zeros((1, 1))
    pl = tensor.Kernel(objectivePressureLoss)(nPlaneCells, (pl,))(U, T, p, cells, areas, w, solver=solver)

    _heatTransfer = tensor.Kernel(objectiveHeatTransfer)
    ht = tensor.Zeros((1, 1))
    for index, patchID in enumerate(patches):
        patch = mesh.boundary[patchID]
        startFace, nFaces = patch['startFace'], patch['nFaces']
        meshArgs = _meshArgs(startFace)
        weight = weights[index]
        ht = _heatTransfer(nFaces, (ht,))(U, T, p, weight, w2, *meshArgs, solver=solver)

    k = solver.mu(300)*solver.Cp/solver.Pr
    a = 0.4
    b = -0.71e-3/(120*k)/2000.

    # MPI ALLREDUCE
    #if not config.gpu:
    #    inputs = (pl, w, ht, w2)
    #    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    #    pl, w, ht, w2 = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs
    inputs = (pl, ht)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    pl, ht = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    # then elemwise
    def _combine(pl, ht):
        pl = pl.scalar()
        ht = ht.scalar()
        obj = pl
        obj2 = ht
        return a*obj + b*obj2
    return tensor.Kernel(_combine)(1)(pl, ht)


