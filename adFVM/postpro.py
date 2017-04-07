import numpy as np

from . import parallel, config
from .field import IOField
from .op import div, grad
from .interp import central
from .compat import intersectPlane

def computeGradients(solver, U, T, p):
    mesh = solver.mesh
    g = solver.gamma
    c = (g*T*solver.R).sqrt()
    ghost = False

    #divU
    UF = central(U, mesh)
    gradU = grad(UF, ghost=ghost)
    #ULF, URF = TVD_dual(U, gradU)
    #UFN = 0.5*(ULF + URF)
    #divU = div(UF.dotN(), ghost=True)
    UFN = UF.dot(mesh.Normals)
    divU = div(UFN, ghost=ghost)

    #speed of sound
    cF = central(c, mesh)
    gradc = grad(cF, ghost=ghost)
    pF = central(p, mesh)
    gradp = grad(pF, ghost=ghost)
    c = c.getInternal()
    p = p.getInternal()
    gradrho = g*(gradp-c*p)/(c*c)

    return gradrho.field, gradU.field, gradp.field, gradc.field, divU.field

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
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
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
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
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
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
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
    phiByV = np.zeros((mesh.origMesh.nCells,) + phi.dimensions, config.precision)
    nInternalCells = mesh.origMesh.nInternalCells
    phiByV[:nInternalCells] = phi.field[:nInternalCells]/mesh.origMesh.volumes
    phiByV = IOField(phi.name + 'ByV', phiByV, (1,), boundary=mesh.calculatedBoundary)
    return phiByV

def getAdjointEnergy(solver, rhoa, rhoUa, rhoEa):
    # J = rhohV*rho/t
    mesh = solver.mesh.origMesh
    Uref, Tref, pref = solver.Uref, solver.Tref, solver.pref
    rhoref = pref/(Tref*solver.R)
    rhoUref = Uref*rhoref
    rhoEref = (solver.Cv*Tref + Uref**2/2)*rhoref
    
    adjEnergy = (rhoref*rhoa.getInternalField()**2*mesh.volumes).sum(axis=1)
    adjEnergy += (rhoUref*rhoUa.getInternalField()**2*mesh.volumes).sum(axis=1)
    adjEnergy += (rhoEref*rhoEa.getInternalField()**2*mesh.volumes).sum(axis=1)
    adjEnergy = (parallel.sum(adjEnergy)**0.5)/(solver.Jref*solver.tref)
    return adjEnergy

def getAdjointMatrixNorm(rhoa, rhoUa, rhoEa, rho, rhoU, rhoE, U, T, p, *outputs, **kwargs):
    mesh = rho.mesh
    solver = rho.solver
    Uref, Tref, pref = solver.Uref, solver.Tref, solver.pref
    g = solver.gamma
    sg = np.sqrt(g)
    g1 = g-1
    sg1 = np.sqrt(g1)
    sge = sg1*sg
    energy = None
    diss = None
    if 'visc' in kwargs:
        visc = kwargs['visc']
    else:
        visc = 'entropy'
    suffix = '_' + visc

    gradrho, gradU, gradp, gradc, divU = outputs
    U = U.getInternal()
    T = T.getInternal()
    p = p.getInternal()
    rho, _, _ = solver.conservative(U, T, p)
    rho = rho.field
    p = p.field
    U = U.field
    c = np.sqrt(g*p/rho)
    b = c/sg
    a = sg1*c/sg
    gradb = gradc/sg
    grada = gradc*sg1/sg
    Z = np.zeros_like(divU)
    
    if visc == 'abarbanel':
        M1 = np.stack((np.hstack((divU, gradb, Z)),
                   np.hstack((gradb[:,[0]], divU, Z, Z, grada[:,[0]])),
                   np.hstack((gradb[:,[1]], Z, divU, Z, grada[:,[1]])),
                   np.hstack((gradb[:,[2]], Z, Z, divU, grada[:,[2]])),
                   np.hstack((Z, grada, divU))),
                   axis=1)
        M2 = np.concatenate((np.hstack((Z, b*gradrho/rho, sg1*divU/2)).reshape(-1,1,5),
                        np.dstack((np.hstack((Z,Z,Z)).reshape(-1,3,1), gradU, (a*gradp/(2*p)).reshape(-1, 3, 1))),
                        np.hstack((Z, 2*grada/g1, g1*divU/2)).reshape(-1,1,5)),
                        axis=1)
        T = np.stack((
                np.hstack((b/rho, Z, Z, Z, Z)),
                np.hstack((-U[:,[0]]/rho, 1/rho, Z, Z, Z)),
                np.hstack((-U[:,[1]]/rho, Z, 1/rho, Z, Z)),
                np.hstack((-U[:,[2]]/rho, Z, Z, 1/rho, Z)),
                np.hstack(((-2*c*c+g*g1*(U*U).sum(axis=1,keepdims=1))/(2*c*sge*rho), -sge*U[:,[0]]/(c*rho), -sge*U[:,[1]]/(c*rho), -sge*U[:,[2]]/(c*rho), sge/(c*rho))),
            ), axis=2)

    # Entropy
    elif visc == 'entropy' or visc == 'uniform':
        M1 = np.stack((np.hstack((divU, gradc, Z)),
                   np.hstack((gradc[:,[0]], divU, Z, Z, Z)),
                   np.hstack((gradc[:,[1]], Z, divU, Z, Z)),
                   np.hstack((gradc[:,[2]], Z, Z, divU, Z)),
                   np.hstack((Z, Z, Z, Z, divU))),
                   axis=1)
        #M2 = np.concatenate((np.hstack((g1*divU/2, gradp/(rho*c), divU/(2*rho*c))).reshape(-1,1,5),
        #                np.dstack(((g1*gradp/(2*rho*c)).reshape(-1,3,1), gradU, (gradp/(2*g*rho*c)).reshape((-1, 3, 1)))),
        #                np.hstack((Z, gradp-c*c*gradrho, Z)).reshape(-1,1,5)),
        #                axis=1)
        #T = np.stack((
        #        np.hstack((b/rho, -g1*U/(c*rho), g1/(c*rho))),
        #        np.hstack((-U[:,[0]]/rho, 1/rho, Z, Z, Z)),
        #        np.hstack((-U[:,[1]]/rho, Z, 1/rho, Z, Z)),
        #        np.hstack((-U[:,[2]]/rho, Z, Z, 1/rho, Z)),
        #        np.hstack((-c*c+g1/2*(U*U).sum(axis=1,keepdims=1), -g1*U, (g-1)*np.ones_like(Z))),
        #    ), axis=2)

        M2 = np.concatenate((np.hstack((g1*divU/2, gradp/(rho*c), divU*pref/(2*rho*c*Uref))).reshape(-1,1,5),
                        np.dstack(((g1*gradp/(2*rho*c)).reshape(-1,3,1), gradU, (gradp*pref/(2*g*rho*c*Uref)).reshape((-1, 3, 1)))),
                        np.hstack((Z, (gradp-c*c*gradrho)*Uref/pref, Z)).reshape(-1,1,5)),
                        axis=1)
        U2 = (U*U).sum(axis=1,keepdims=1)
        T = np.stack((
                np.hstack((g1*U2/(2*c*rho*Uref), -g1*U/(c*rho*Uref), g1/(c*rho*Uref))),
                np.hstack((-U[:,[0]]/(rho*Uref), 1/(rho*Uref), Z, Z, Z)),
                np.hstack((-U[:,[1]]/(rho*Uref), Z, 1/(rho*Uref), Z, Z)),
                np.hstack((-U[:,[2]]/(rho*Uref), Z, Z, 1/(rho*Uref), Z)),
                np.hstack(((-2*g*p+g1*rho*U2)/(2*pref*rho), -g1*U/pref, (g-1)/pref*np.ones_like(Z))),
            ), axis=2)

        #suffix += '_factor'

    M = M1/2-M2
    #MS = (M + M.transpose((0, 2, 1)))/2
    #M_2norm = np.linalg.eigvalsh(MS)[:,[-1]]
    #M_2norm = IOField('M_2norm_old' + suffix, M_2norm, (1,), boundary=mesh.calculatedBoundary)
    #M_2norm.write()
    
    def dot(a, b):
        return np.sum(a*b.reshape(-1,1,5), axis=-1)
    Ti = np.linalg.inv(T)

    X = np.diag([1, 1./Uref, 1./Uref, 1./Uref, 1/pref]).reshape(1,5,5)
    TiX = np.matmul(Ti, X)
    Mc = np.matmul(TiX.transpose(0, 2, 1), np.matmul(M, TiX))
    MS = (Mc + Mc.transpose((0, 2, 1)))/2
    M_2norm = np.linalg.eigvalsh(MS)[:,[-1]]

    M_2norm /= parallel.max(M_2norm)
    M_2norm = 1/(1+np.exp(-5*(M_2norm-1)))
    suffix += '_factor'

    #print parallel.min(M_2norm)
    #parallel.pprint(parallel.max(M_2norm))
    #parallel.pprint(parallel.min(M_2norm))

    #M_2norm /= np.sqrt(parallel.sum(M_2norm**2*mesh.volumes)/parallel.sum(mesh.volumes))
    #parallel.pprint(parallel.max(M_2norm))
    #parallel.pprint(parallel.min(M_2norm))
    if visc == "uniform":
        M_2norm = np.ones_like(M_2norm)
    M_2norm = IOField('M_2norm' + suffix, M_2norm, (1,))
    #M_2norm.write()
    
    if rhoa is not None:
        w = np.hstack((rhoa.field[:mesh.nInternalCells], rhoUa.field[:mesh.nInternalCells], rhoEa.field[:mesh.nInternalCells]))
        v = dot(Ti, w)
        energy = np.sum(dot(M, v)*v, axis=-1,keepdims=1)
        energy = np.maximum(0, energy)
        energy = IOField('energy' + suffix, 1e-30*energy, (1,))

        Vs = parallel.sum(mesh.volumes)
        rhoa.defaultComplete()
        rhoUa.defaultComplete()
        rhoEa.defaultComplete()
        gradrhoa = grad(central(rhoa, mesh))
        gradrhoUa = grad(central(rhoUa, mesh))
        gradrhoEa = grad(central(rhoEa, mesh))
        gradw = np.dstack((gradrhoa.field, gradrhoUa.field, gradrhoEa.field))
        gradw /= np.diag(X[0]).reshape(1,1,5)
        gradwN = (gradw**2).sum(axis=(1,2)).reshape(-1,1)
        diss = M_2norm.field*gradwN
        diss = IOField('diss' + suffix, 1e-30*diss, (1,))

        dissn = np.sqrt(parallel.sum(diss.field**2*mesh.volumes)/Vs)
        en = np.sqrt(parallel.sum(energy.field**2*mesh.volumes)/Vs)
        corr = parallel.sum(diss.field*energy.field*mesh.volumes)/(Vs*dissn*en)
        parallel.pprint('energy and M_2norm corr:', corr)

        diss.defaultComplete()
        energy.defaultComplete()

    M_2norm.defaultComplete()

    return M_2norm, energy, diss


def getAdjointViscosity(rho, rhoU, rhoE, scaling, outputs=None, init=True):
    solver = rho.solver
    mesh = rho.mesh
    if init:
        rho, rhoU, rhoE = solver.initFields((rho, rhoU, rhoE))
    U, T, p = solver.primitive(rho, rhoU, rhoE)

    if not outputs:
        outputs = computeGradients(solver, U, T, p)
    M_2norm, _ = getAdjointMatrixNorm(None, None, None, rho, rhoU, rhoE, U, T, p, *outputs)
    viscosity = M_2norm*float(scaling)
    viscosity.name = 'mua'
    viscosity.boundary = mesh.calculatedBoundary
    return viscosity
