import numpy as np

from adFVM import parallel
from adFVM.field import IOField, CellField
from adFVM.op import div, grad
from adFVM.interp import central
from adFVM.compat import intersectPlane
from adFVM.config import ad

def computeGradients(solver):
    mesh = solver.mesh
    g = solver.gamma
    p, U, T = solver.symbolicFields()
    c = (g*T*solver.R).sqrt()

    #divU
    UF = central(U, mesh)
    gradU = grad(UF, ghost=True)
    #ULF, URF = TVD_dual(U, gradU)
    #UFN = 0.5*(ULF + URF)
    #divU = div(UF.dotN(), ghost=True)
    divU = div(UF.dot(mesh.Normals), ghost=True)

    #speed of sound
    gradc = grad(central(c, mesh), ghost=True)
    gradp = grad(central(p, mesh), ghost=True)
    gradrho = g*(gradp-c*p)/(c*c)

    computer = solver.function([U.field, T.field, p.field], [gradrho.field, 
                                      gradU.field, 
                                      gradp.field, 
                                      gradc.field, 
                                      divU.field
                                     ], 'compute')
    return computer

def getEnstrophyAndQ(gradU):
    enstrophy =  gradU.norm()
    gradUT = gradU.transpose()
    omega = 0.5*(gradU - gradUT)
    S = 0.5*(gradU + gradUT)
    Q = 0.5*(omega.norm()**2 - S.norm()**2)
    return enstrophy, Q

def getYPlus(U, T, rho, patches):
    mesh = U.mesh.origMesh
    solver = U.solver
    yplus = {}
    uplus = {}
    ustar = {}
    yplus1 = {}
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
        internalIndices = mesh.owner[startFace:endFace]
        faceIndices = mesh.neighbour[startFace:endFace]
        deltas = mesh.deltas[startFace:endFace]

        Uw = U.field[faceIndices]
        Ui = U.field[internalIndices]
        Tw = T.field[faceIndices]
        rhow = rho.field[faceIndices]
        nuw = solver.mu(Tw)/rhow
        tauw = nuw*(Ui-Uw)/deltas
        tauw = (tauw**2).sum(axis=1, keepdims=True)**0.5
        ustar[patchID] = tauw**0.5
        yplus1[patchID] = nuw/ustar[patchID]
        yplus[patchID] = deltas/yplus1[patchID]
        uplus[patchID] = Ui/ustar[patchID]
 
    return uplus, yplus, ustar, yplus1

def getHTC(T, T0, patches):
    mesh = T.mesh.origMesh
    solver = T.solver
    htc = {}
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
        internalIndices = mesh.owner[startFace:endFace]
        deltas = mesh.deltas[startFace:endFace]

        Ti = T.field[internalIndices] 
        Tw = T.field[mesh.neighbour[startFace:endFace]]
        dtdn = (Ti-Tw)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        dT = T0-Tw
        htc[patchID] = k*dtdn/dT
    return htc

def getIsentropicMa(p, p0, patches):
    mesh = p.mesh.origMesh
    solver = p.solver
    g = solver.gamma
    Ma = {}
    for patchID in patches:
        startFace = mesh.boundary[patchID]['startFace']
        nFaces = mesh.boundary[patchID]['nFaces']
        endFace = startFace + nFaces
        pw = p.field[mesh.neighbour[startFace:endFace]]
        Ma[patchID] = (2.0/(g-1)*((1./p0*pw)**((1-g)/g)-1))**0.5
    return Ma

def getTotalPressureAndEntropy(U, T, p, solver):
    g = solver.gamma
    c = (g*solver.R*T).sqrt()
    rho = p/(solver.R*T)
    M = U.mag()/c
    pt = p*(1+0.5*(g-1)*M**2)**(g/(g-1))
    s = solver.Cv*(p*rho**-g).log()
    return c, M, pt, s

def getPressureLoss(p, T, U, p0, point, normal):
    solver = p.solver
    mesh = p.mesh.origMesh

    cells, _ = intersectPlane(solver.mesh, point, normal)

    g = solver.gamma
    pi, Ti, Ui = p.field[cells], T.field[cells], U.field[cells]
    ci = (g*solver.R*Ti)**0.5
    Umagi = (Ui*Ui).sum(axis=1, keepdims=True)**0.5
    Mi = Umagi/ci
    pti = pi*(1 + 0.5*(g-1)*Mi*Mi)**(g/(g-1))
    return cells, p0 - pti

def getFieldByVolume(phi):
    mesh = phi.mesh
    phiByV = np.zeros((mesh.origMesh.nCells,) + phi.dimensions)
    nInternalCells = mesh.origMesh.nInternalCells
    phiByV[:nInternalCells] = phi.field[:nInternalCells]/mesh.origMesh.volumes
    phiByV = IOField(phi.name + 'ByV', phiByV, (1,), boundary=mesh.calculatedBoundary)
    return phiByV

def getAdjointEnergy(solver, rhoa, rhoUa, rhoEa):
    mesh = rhoa.mesh
    # J = rhohV*rho/t
    Uref, Tref, pref = solver.Uref, solver.Tref, solver.pref
    rhoref = pref/(Tref*solver.R)
    rhoUref = Uref*rhoref
    rhoEref = (solver.Cv*Tref + Uref**2/2)*rhoref
    adjEnergy = (rhoref*rhoa.getInternalField()**2).sum(axis=1)
    adjEnergy += (rhoUref*rhoUa.getInternalField()**2).sum(axis=1)
    adjEnergy += (rhoEref*rhoEa.getInternalField()**2).sum(axis=1)
    adjEnergy = (parallel.sum(adjEnergy)**0.5)/(solver.Jref*solver.tref)
    return adjEnergy

def getAdjointNorm(rho, rhoU, rhoE, U, T, p, *outputs):
    mesh = rho.mesh
    solver = rho.solver
    g = solver.gamma
    sg = np.sqrt(g)
    g1 = g-1
    sg1 = np.sqrt(g1)

    gradrho, gradU, gradp, gradc, divU = outputs
    rho = rho.field
    p = p.field
    c = np.sqrt(g*p/rho)
    b = c/sg
    a = sg1*c/sg
    gradb = gradc/sg
    grada = gradc*sg1/sg
    Z = np.zeros_like(divU)
    Z3 = np.zeros_like(gradU)
    
    M1 = np.dstack((np.hstack((divU, gradb, Z)),
               np.hstack((gradb[:,[0]], divU, Z, Z, grada[:,[0]])),
               np.hstack((gradb[:,[1]], Z, divU, Z, grada[:,[1]])),
               np.hstack((gradb[:,[2]], Z, Z, divU, grada[:,[2]])),
               np.hstack((Z, grada, divU))))

    M2 = np.dstack((np.hstack((Z, b*gradrho/rho, sg1*divU/2)),
                    np.hstack((np.dstack((Z,Z,Z)), gradU, (a*gradp/(2*p)).reshape(-1, 1, 3))),
                    np.hstack((Z, 2*grada/g1, g1*divU/2))))
    M = M1-M2

    #U, S, V = np.linalg.svd(M)
    #M_2norm = np.ascontiguousarray(S[:, [0]])
    MS = (M + M.transpose((0, 2, 1)))/2
    M_2norm = np.linalg.eigvalsh(MS)[:,[-1]]
    #M_2norm = np.linalg.eigh(MS, eigvals=(4), eigvals_only=True).reshape(-1,1)

    M_2norm = IOField('M_2norm', M_2norm, (1,), boundary=mesh.calculatedBoundary)
    return [M_2norm]
