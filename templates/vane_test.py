import numpy as np

from adFVM import config
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 
from adpy import tensor

#primal = RCF('/home/talnikar/adFVM/cases/vane_optim/foam/laminar/3d_baseline/par-16/', objective='drag', objectiveDragInfo='pressure')
#primal = RCF('/master/home/talnikar/adFVM/cases/vane/les/', faceReconstructor='SecondOrder')#, timeIntegrator='euler')
#primal = RCF('/master/home/talnikar/foam/blade/les/')
#primal = RCF('/lustre/atlas/proj-shared/tur103/les/')

def dot(a, b):
    return ad.reshape(ad.sum(a*b, axis=1), (-1,1))

# drag over cylinder surface
def objectiveDrag(solver, mesh):
    U, T, p = tensor.CellTensor((3,)), tensor.CellTensor((1,)), tensor.CellTensor((1,))
    U0 = U.extract(mesh.neighbour)[0]
    U0i = U.extract(mesh.owner)[0]
    p0 = p.extract(mesh.neighbour)
    T0 = T.extract(mesh.neighbour)
    nx = mesh.normals[0]
    mungUx = solver.mu(T0)*(U0-U0i)/mesh.deltas
    drag = (p0*nx-mungUx)*mesh.areas
    return tensor.TensorFunction('objective', [U, T, p, mesh.areas, mesh.deltas, mesh.normals,
        mesh.owner, mesh.neighbour], [drag])

# heat transfer
def objectiveHeatTransfer(solver, mesh):
    U, T, p = tensor.CellTensor((3,)), tensor.CellTensor((1,)), tensor.CellTensor((1,))
    Ti = T.extract(mesh.owner)
    Tw = 300.
    dtdn = (Tw-Ti)/mesh.deltas
    k = solver.Cp*solver.mu(Tw)/solver.Pr
    ht = k*dtdn*mesh.areas
    w = mesh.areas*1.
    return tensor.TensorFunction('objective2', [U, T, p, mesh.areas, mesh.deltas, mesh.owner], [ht, w])

# pressure loss
def getPlane(solver):
    point = np.array([0.052641,-0.1,0.005])
    normal = np.array([1.,0.,0.])
    interCells, interArea = intersectPlane(solver.mesh, point, normal)
    return {'cells':interCells.astype(np.int32), 
            'areas': interArea, 
           }

def objectivePressureLoss(solver, mesh):
    ptin = 175158.
    normal = np.array([1.,0.,0.])
    U, T, p = tensor.CellTensor((3,)), tensor.CellTensor((1,)), tensor.CellTensor((1,))
    areas = tensor.Tensor((1,))
    cells = tensor.Tensor((1,), [tensor.IntegerScalar()])
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
    return tensor.TensorFunction('objective', [U, T, p, areas, cells], [pl, w])
    
#objective = objectiveHeatTransfer
objective = objectiveDrag

objectiveString = """
scalar objective(const mat& U, const vec& T, const vec& p) {{
    const Mesh& mesh = *meshp;
    string patchID = "{0}";
    integer startFace, nFaces;
    tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
    vec drag(nFaces, true);
    Function_objective(nFaces, &U(0), &T(0), &p(0), \
        &mesh.areas(startFace), &mesh.deltas(startFace), &mesh.normals(startFace), &mesh.owner(startFace), &mesh.neighbour(startFace), \
        &drag(0));
    scalar d = drag.sum();
    scalar gd = 0;
    MPI_Allreduce(&d, &gd, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return gd;
}}
void objective_grad(const mat& U, const vec& T, const vec& p, mat& Ua, vec& Ta, vec& pa) {{
    const Mesh& mesh = *meshp;
    Mesh& meshAdj = *meshap;
    string patchID = "{0}";
    integer startFace, nFaces;
    tie(startFace, nFaces) = mesh.boundaryFaces.at(patchID);
    vec draga(nFaces);
    for (integer i = 0; i < nFaces; i++) {{
        draga(i) = 1;
    }}
    Function_objective_grad(nFaces, &U(0), &T(0), &p(0), \
        &mesh.areas(startFace), &mesh.deltas(startFace), &mesh.normals(startFace), &mesh.owner(startFace), &mesh.neighbour(startFace), \
        &draga(0), \
        &Ua(0), &Ta(0), &pa(0),
        &meshAdj.areas(startFace), &meshAdj.deltas(startFace), &meshAdj.normals(startFace), &meshAdj.owner(startFace), &meshAdj.neighbour(startFace)); \
}}
""".format('pressure')


primal = RCF('/home/talnikar/adFVM/cases/vane/laminar/', objective=objective, \
             objectiveString = objectiveString)

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
perturb = [makePerturb(1)]

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
nSteps = 10
writeInterval = 5
#nSteps = 100000
#writeInterval = 5000
startTime = 3.0
dt = 1e-8

