import numpy as np

from adFVM import config
from adFVM.config import ad
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 

#primal = RCF('/home/talnikar/adFVM/cases/vane_optim/foam/laminar/3d_baseline/par-16/', objective='drag', objectiveDragInfo='pressure')
#primal = RCF('/master/home/talnikar/adFVM/cases/vane/les/', faceReconstructor='SecondOrder')#, timeIntegrator='euler')
#primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')

def dot(a, b):
    return ad.reshape(ad.sum(a*b, axis=1), (-1,1))

# heat transfer
def objectiveHeatTransfer(solver, mesh):
    U, T, p = tensor.CellTensor((3,)), tensor.CellTensor((1,)), tensor.CellTensor((1,))
    Ti = T.extract(mesh.owner)
    Tw = 300.
    dtdn = (Tw-Ti)/mesh.deltas
    k = solver.Cp*solver.mu(Tw)/solver.Pr
    ht = k*dtdn*mesh.areas
    w = mesh.areas
    return tensor.TensorFunction('objective2', [U, T, p, mesh.areas, mesh.deltas, mesh.owner], [ht, w])

# pressure loss
def getPlane(solver):
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    return {'cells':interCells.astype(np.int32), 
            'areas': interArea, 
            'ptin': ptin
           }

def objectivePressureLoss(solver, mesh):
    ptin = 175158.
    normal = np.array([1.,0.,0.])
    U, T, p = tensor.CellTensor((3,)), tensor.CellTensor((1,)), tensor.CellTensor((1,))
    areas = Tensor((1,))
    cells = Tensor((1,), IntegerScalar())
    g = solver.gamma
    pi = p.extract(cells)
    Ti = T.extract(cells)
    Ui = U.extract(cells)
    rhoi = pi/(solver.Cv*Ti*(g- 1))
    ci = (gpi/rhoi).sqrt()

    rhoUni = sum([rhoi*Ui[i]*normal[i] for i in range(0, 3)])
    Umagi = Ui.dot(Ui)
    Mi = Umagi.sqrt()/ci
    pti = pi*pow(1 + 0.5*(g-1)*Mi*Mi, g/(g-1))
    pl = (ptin-pti)*rhoUni*areas/ptin
    w = rhoUni*areas
    return tensor.TensorFunction('objective', [U, T, p, areas, cells], [pl, w])
    
#objective = objectiveHeatTransfer
objective = objectivePressureLoss

objectiveString = """
scalar objective(const mat& U, const vec& T, const vec& p) {{
    integer nCells = rcf->objectivePLInfo["cells"].size()/sizeof(integer);
    integer* cells = (integer*) rcf->objectivePLInfo.at("cells").data();
    scalar* areas = (uscalar*) rcf->objectivePLInfo.at("areas").data();
    const Mesh& mesh = *meshp;
    vec loss(nCells, true);
    vec weights(nCells, true);
    Function_objective(nCells, &U(0), &T(0), &p(0), \
        areas, cells, \
        &loss(0), &weights(0));
    scalar pl = loss.sum();
    scalar w = weights.sum();
    scalar gpl = 0, gw = 0;
    MPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pl, &gpl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    scalar obj = gpl/gw;

    scalar ht = 0;
    w = 0;
    for (string patchID : split({0}, '|')) {{
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        vec heat(nFaces, true);
        vec weights(nFaces, true);
        Function_objective(nCells, &U(0), &T(0), &p(0), \
            &mesh.areas(startFace), &mesh.deltas(startFace), &mesh.owner(startFace), \
            &heat(0), &weights(0));
        ht += heat.sum();
        w += weights.sum();
    }}
    MPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ht, &ght, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    scalar obj2 = ght/gw;

    scalar a = {1};
    scalar b = {2};
    return a*obj + b*obj2;
}}
void objective_grad(const mat& U, const vec& T, const vec& p, mat& Ua, vec& Ta, vec& pa) {{
    scalar a = {1};
    scalar b = {2};

    integer nCells = rcf->objectivePLInfo["cells"].size()/sizeof(integer);
    integer* cells = (integer*) rcf->objectivePLInfo.at("cells").data();
    scalar* areas = (uscalar*) rcf->objectivePLInfo.at("areas").data();
    const Mesh& mesh = *meshp;
    vec loss(nCells, true);
    vec weights(nCells, true);
    Function_objective(nFaces, &U(0), &T(0), &p(0), \
        areas, cells, \
        &loss(0), &weights(0));
    scalar pl = loss.sum();
    scalar w = weights.sum();
    scalar gpl = 0, gw = 0;
    MPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&pl, &gpl, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
        &areasa, NULL);

    scalar ht = 0;
    w = 0;
    for (string patchID : split({0}, '|')) {{
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        vec heat(nFaces, true);
        vec weights(nFaces, true);
        Function_objective2(nFaces, &U(0), &T(0), &p(0), \
            &mesh.areas(startFace), &mesh.deltas(startFace), &mesh.owner(startFace), \
            &heat(0), &weights(0));
        ht += heat.sum();
        w += weights.sum();
    }}
    MPI_Allreduce(&w, &gw, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ht, &ght, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (string patchID : split({0}, '|')) {{
        integer startFace, nFaces;
        tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
        vec heata(nFaces, true);
        vec weightsa(nFaces, true);
        for (integer i = 0; i < nFaces; i++) {{
            lossa(i) = b/gw;
            weightsa(i) = -b*ght/(gw*gw);
        }}
        Function_objective2_grad(nFaces, &U(0), &T(0), &p(0), \
            &mesh.areas(startFace), &mesh.deltas(startFace), &mesh.owner(startFace), \
            &heata(0), &weightsa(0), \
            &Ua(0), &Ta(0), &pa(0), \
            &meshAdj.areas(startFace), &meshAdj.deltas(startFace), &meshAdj.owner(startFace) \
        );
    }}
}}
"""

primal = RCF('/home/talnikar/adFVM/cases/vane/laminar/', objective=[objectivePressureLoss, objectiveHeatTransfer], objectivePLInfo={}, \
             objectiveString = objectiveString)
primal.defaultConfig["objectivePLInfo"] = getPlane(primal)
k = solver.mu(300)*solver.Cp/solver.Pr
b = -0.71e-3/(120*k)/2000.
primal.objectiveString = primal.objectiveString.format('pressure|suction', 0.4, b)

def makePerturb(param, eps=1e-6):
    def perturbMesh(fields, mesh, t):
        if not hasattr(perturbMesh, 'perturbation'):
            ## do the perturbation based on param and eps
            #perturbMesh.perturbation = mesh.getPerturbation()
            points = np.zeros_like(mesh.parent.points)
            points[param] = eps
            perturbMesh.perturbation = mesh.parent.getPointsPerturbation(points)
        return perturbMesh.perturbation
    return perturbMesh
perturb = [makePerturb(1), makePerturb(2)]

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

#nSteps = 10
#writeInterval = 5
nSteps = 20000
writeInterval = 500
#nSteps = 100000
#writeInterval = 5000
startTime = 1.0
dt = 1e-8

