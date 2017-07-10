import numpy as np

from adFVM import config
from adFVM.compat import norm, intersectPlane
from adFVM.density import RCF 
from adFVM import tensor

def dot(a, b):
    return ad.sum(a*b, axis=1, keepdims=True)

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
""".format('cylinder')

#primal = RCF('cases/cylinder_steady/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*5e-5, (1,)))
#primal = RCF('cases/cylinder_per/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*5e-5, (1,)))
#primal = RCF('cases/cylinder_chaos_test/', CFL=1.2, mu=lambda T: Field('mu', T.field/T.field*2.5e-5, (1,)), boundaryRiemannSolver='eulerLaxFriedrichs')
primal = RCF('/home/talnikar/adFVM/cases/cylinder/',
#primal = RCF('/home/talnikar/adFVM/cases/cylinder/Re_500/',
#primal = RCF('/home/talnikar/adFVM/cases/cylinder/chaotic/testing/', 
             timeIntegrator='SSPRK', 
             CFL=1.2, 
             #mu=lambda T: 2.5e-5*T/T,
             mu=lambda T: 2.5e-5,
             faceReconstructor='SecondOrder',
             boundaryRiemannSolver='eulerLaxFriedrichs',
             objective = objective,
             objectiveString = objectiveString
)



def makePerturb(scale):
    def perturb(fields, mesh, t):
        #mid = np.array([-0.012, 0.0, 0.])
        #G = 100*np.exp(-3e4*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
        mid = np.array([-0.0005, 0.0, 0.])
        #G = scale*np.exp(-2.5e9*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
        G = scale*np.exp(-2.5e6*norm(mid-mesh.cellCentres[:mesh.nInternalCells], axis=1)**2)
        rho = G
        rhoU = np.zeros((mesh.nInternalCells, 3))
        rhoU[:, 0] += G.flatten()*100
        rhoE = G*2e5
        return rho, rhoU, rhoE
    return perturb
 
perturb = [makePerturb(1e6)]
parameters = 'source'

#def makePerturb(pt_per):
#    def perturb(fields, mesh, t):
#        return pt_per
#    return perturb

#perturb = [makePerturb(0.1), makePerturb(0.2), makePerturb(0.4)]
#perturb = [makePerturb(1.)]
#parameters = ('BCs', 'p', 'left', 'U0')

nSteps = 10
writeInterval = 2
reportInterval = 1
startTime = 3.0
dt = 1e-8
#adjParams = [1e-3, 'abarbanel', None]
runCheckpoints = 3
