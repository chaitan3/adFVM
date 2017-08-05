import numpy as np

from adFVM import config
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 
from adFVM import tensor
from adFVM.mesh import Mesh

#primal = RCF('/home/talnikar/adFVM/cases/vane_optim/foam/laminar/3d_baseline/par-16/', objective='drag', objectiveDragInfo='pressure')
#primal = RCF('/master/home/talnikar/adFVM/cases/vane/les/', faceReconstructor='SecondOrder')#, timeIntegrator='euler')
#primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')

def dot(a, b):
    return ad.reshape(ad.sum(a*b, axis=1), (-1,1))

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
def getPlane(solver):
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
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
        solver.extraArgs.append((tensor.Variable((nFaces, 1)), weights*1.))
    
def objective(fields, solver):
    U, T, p = fields
    mesh = solver.mesh.symMesh
    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    nPlaneCells, cells, areas = [x[0] for x in solver.extraArgs[:3]]
    pl, w = tensor.Zeros((1, 1)), tensor.Zeros((1, 1))
    pl, w = tensor.Tensorize(objectivePressureLoss)(nPlaneCells, (pl, w))(U, T, p, cells, areas, solver=solver)

    _heatTransfer = tensor.Tensorize(objectiveHeatTransfer)
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
    inputs = (pl, w, ht, w2)
    outputs = tuple([tensor.Zeros(x.shape) for x in inputs])
    pl, w, ht, w2 = tensor.ExternalFunctionOp('mpi_allreduce', inputs, outputs).outputs

    # then elemwise
    def _combine(pl, w, ht, w2):
        obj = pl/w
        obj2 = ht/w2
        return a*obj + b*obj2
    return tensor.Tensorize(_combine)(1)(pl, w, ht, w2)[0]

objectiveString = """
scalar objective(const mat& U, const vec& T, const vec& p) {{
    integer nCells = rcf->objectivePLInfo["cells"].size()/sizeof(integer);
    integer* cells = (integer*) rcf->objectivePLInfo.at("cells").data();
    scalar* areas = (scalar*) rcf->objectivePLInfo.at("areas").data();
    const Mesh& mesh = *meshp;
    vec loss(nCells, true);
    vec weights(nCells, true);
    Function_objective(nCells, &U(0), &T(0), &p(0), \
        areas, cells, \
        &loss(0), &weights(0));
    scalar pl = loss.sum();
    scalar w = weights.sum();

    scalar ht = 0;
    scalar w2 = 0;
    vector<string> patches = {{"{0}", "{3}"}};
    for (string patchID : patches) {{
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        vec heat(nFaces, true);
        vec weights(nFaces, true);
        for (integer f = startFace; f < startFace + nFaces; f++) {{
            if (patchID == "pressure") {{
                if ((mesh.faceCentres(f, 0) < 0.33757) || (mesh.faceCentres(f, 1) > 0.04692))
                continue;
            }}
            if (patchID == "suction") {{
                if ((mesh.faceCentres(f, 0) < 0.035241) || (mesh.faceCentres(f, 1) > 0.044337))
                continue;
            }}
            Function_objective2(1, &U(0), &T(0), &p(0), \
                &mesh.areas(f), &mesh.deltas(f), &mesh.owner(f), \
                &heat(f-startFace), &weights(f-startFace));
        }}
        ht += heat.sum();
        w2 += weights.sum();
    }}

    scalar val[4] = {{pl, w, ht, w2}};
    scalar gval[4];
    MPI_Allreduce(&val, &gval, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    scalar a = {1};
    scalar b = {2};
    scalar obj = gval[0]/gval[1];
    scalar obj2 = gval[2]/gval[3];
    //cout << a*obj << " " << b*obj2 << endl;
    return a*obj + b*obj2;
}}
void objective_grad(const mat& U, const vec& T, const vec& p, mat& Ua, vec& Ta, vec& pa) {{
    const Mesh& mesh = *meshp;
    Mesh& meshAdj = *meshap;

    integer nCells = rcf->objectivePLInfo["cells"].size()/sizeof(integer);
    integer* cells = (integer*) rcf->objectivePLInfo.at("cells").data();
    scalar* areas = (scalar*) rcf->objectivePLInfo.at("areas").data();
    vec loss(nCells, true);
    vec weights(nCells, true);
    Function_objective(nCells, &U(0), &T(0), &p(0), \
        areas, cells, \
        &loss(0), &weights(0));
    scalar pl = loss.sum();
    scalar w = weights.sum();

    scalar ht = 0;
    scalar w2 = 0;
    vector<string> patches = {{"{0}", "{3}"}};
    for (string patchID : patches) {{
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        vec heat(nFaces, true);
        vec weights(nFaces, true);
        for (integer f = startFace; f < startFace + nFaces; f++) {{
            if (patchID == "pressure") {{
                if ((mesh.faceCentres(f, 0) < 0.33757) || (mesh.faceCentres(f, 1) > 0.04692))
                continue;
            }}
            if (patchID == "suction") {{
                if ((mesh.faceCentres(f, 0) < 0.035241) || (mesh.faceCentres(f, 1) > 0.044337))
                continue;
            }}
            Function_objective2(1, &U(0), &T(0), &p(0), \
                &mesh.areas(f), &mesh.deltas(f), &mesh.owner(f), \
                &heat(f-startFace), &weights(f-startFace));
        }}
        ht += heat.sum();
        w2 += weights.sum();
    }}

    scalar val[4] = {{pl, w, ht, w2}};
    scalar gval[4];
    MPI_Allreduce(&val, &gval, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    scalar gpl = gval[0];
    scalar gw = gval[1];
    scalar ght = gval[2];
    scalar gw2 = gval[3];
    scalar a = {1};
    scalar b = {2};

    vec lossa(nCells);
    vec weightsa(nCells);
    vec areasa(nCells);
    for (integer i = 0; i < nCells; i++) {{
        lossa(i) = a/gw;
        weightsa(i) = -a*gpl/(gw*gw);
    }}
    Function_objective_grad(nCells, &U(0), &T(0), &p(0), \
        areas, cells, \
        &lossa(0), &weightsa(0),\
        &Ua(0), &Ta(0), &pa(0), \
        &areasa(0), NULL);

    for (string patchID : patches) {{
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        vec heata(nFaces);
        vec weightsa(nFaces);
        for (integer i = 0; i < nFaces; i++) {{
            heata(i) = b/gw;
            weightsa(i) = -b*ght/(gw*gw);
        }}
        for (integer f = startFace; f < startFace + nFaces; f++) {{
            if (patchID == "pressure") {{
                if ((mesh.faceCentres(f, 0) < 0.33757) || (mesh.faceCentres(f, 1) > 0.04692))
                continue;
            }}
            if (patchID == "suction") {{
                if ((mesh.faceCentres(f, 0) < 0.035241) || (mesh.faceCentres(f, 1) > 0.044337))
                continue;
            }}
            Function_objective2_grad(1, &U(0), &T(0), &p(0), \
                &mesh.areas(f), &mesh.deltas(f), &mesh.owner(f), \
                &heata(f-startFace), &weightsa(f-startFace),
                &Ua(0), &Ta(0), &pa(0), \
                &meshAdj.areas(f), &meshAdj.deltas(f), &meshAdj.owner(f));
        }}
    }}
}}
"""

primal = RCF('/home/talnikar/adFVM/cases/vane/laminar/', objective=objective)
getPlane(primal)
getWeights(primal)

def makePerturb(param, eps=1e-4):
    def perturbMesh(fields, mesh, t):
        if not hasattr(perturbMesh, 'perturbation'):
            ## do the perturbation based on param and eps
            #perturbMesh.perturbation = mesh.getPerturbation()
            points = np.zeros_like(mesh.points)
            #points[param] = eps
            points[:] = eps*mesh.points
            perturbMesh.perturbation = mesh.getPointsPerturbation(points)
        return perturbMesh.perturbation
    return perturbMesh
#perturb = [makePerturb(1)]
perturb = []

parameters = 'mesh'

#def makePerturb(mid):
#    def perturb(fields, mesh, t):
#        G = 10*np.exp(-1e4*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
#        #rho
#        rho = G
#        rhoU = np.zeros((mesh.nInternalCells, 3))
#        rhoU[:, 0] = G.flatten()*100
#        rhoE = G*2e5
#        return rho, rhoU, rhoE
#    return perturb
#perturb = [makePerturb(np.array([-0.02, 0.01, 0.005])),
#           makePerturb(np.array([-0.08, -0.01, 0.005]))]

#parameters = 'source'

nSteps = 10
writeInterval = 5
#nSteps = 100
#writeInterval = 20
#nSteps = 100000
#writeInterval = 5000
startTime = 3.0
dt = 1e-8

#adjParams = [1e-3, 'abarbanel', None]
