import numpy as np

from . import parallel, config
from .field import IOField, CellField, Field
from . import op, interp
from .compat import intersectPlane
import time
from .mesh import Mesh
from .parallel import pprint
from adpy.tensor import Tensor, Kernel, ConstantOp, ExternalFunctionOp
from adpy.variable import Variable, Zeros

def computeGradients(solver, U, T, p):
    mesh = solver.mesh
    g = solver.gamma
    c = (g*T*solver.R).sqrt()
    ghost = False

    #divU
    UF = interp.centralOld(U, mesh)
    gradU = op.gradOld(UF, ghost=ghost)
    #ULF, URF = TVD_dual(U, gradU)
    #UFN = 0.5*(ULF + URF)
    #divU = div(UF.dotN(), ghost=True)
    UFN = UF.dot(Field('N', mesh.normals, (3,)))
    divU = op.divOld(UFN, ghost=ghost)

    #speed of sound
    cF = interp.centralOld(c, mesh)
    gradc = op.gradOld(cF, ghost=ghost)
    pF = interp.centralOld(p, mesh)
    gradp = op.gradOld(pF, ghost=ghost)
    c = c.getInternal()
    p = p.getInternal()
    gradrho = g*(gradp-c*p)/(c*c)
    TF = interp.centralOld(T, mesh)
    gradT = op.gradOld(TF)

    return gradT.field, gradrho.field, gradU.field, gradp.field, gradc.field, divU.field

def getEnstrophyAndQ(gradU):
    enstrophy =  gradU.norm()
    gradUT = gradU.transpose()
    omega = 0.5*(gradU - gradUT)
    S = 0.5*(gradU + gradUT)
    Q = 0.5*(omega.norm()**2 - S.norm()**2)
    return enstrophy, Q

def getYPlus(U, T, rho, patches):
    mesh = U.mesh
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
    mesh = T.mesh
    solver = T.solver
    htc = {}
    for patchID in patches:
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
        internalIndices = mesh.owner[startFace:endFace]
        N = mesh.normals[startFace:endFace]
        Deltas = mesh.deltas[startFace:endFace]*mesh.deltasUnit[startFace:endFace]
        deltas = -(Deltas*N).sum(axis=1, keepdims=1)

        Ti = T.field[internalIndices] 
        Tw = T.field[mesh.neighbour[startFace:endFace]]
        dtdn = (Ti-Tw)/deltas
        k = solver.Cp*solver.mu(Tw)/solver.Pr
        dT = T0-Tw
        htc[patchID] = k*dtdn/dT
    return htc

def getIsentropicMa(p, p0, patches):
    mesh = p.mesh
    solver = p.solver
    g = solver.gamma
    Ma = {}
    for patchID in patches:
        startFace, endFace, _ = mesh.getPatchFaceRange(patchID)
        pw = p.field[mesh.neighbour[startFace:endFace]]
        Ma[patchID] = (2.0/(g-1)*((1./p0*pw)**((1-g)/g)-1))**0.5
    return Ma

def getRe(U, T, p, rho, D):
    solver = T.solver
    mu = solver.mu(T)
    Re = U.mag()*rho*D/mu
    return Re

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
    phiByV = np.zeros((mesh.nCells,) + phi.dimensions, config.precision)
    nInternalCells = mesh.nInternalCells
    phiByV[:nInternalCells] = phi.field[:nInternalCells]/mesh.volumes
    phiByV = IOField(phi.name + 'ByV', phiByV, (1,), boundary=mesh.calculatedBoundary)
    return phiByV

def getAdjointEnergy(solver, rhoa, rhoUa, rhoEa):
    # J = rhohV*rho/t
    mesh = solver.mesh
    Uref, Tref, pref = solver.Uref, solver.Tref, solver.pref
    rhoref = pref/(Tref*solver.R)
    rhoUref = Uref*rhoref
    rhoEref = (solver.Cv*Tref + Uref**2/2)*rhoref
    
    # already divided by volumes
    #adjEnergy = (rhoref*rhoa.getInternalField()**2*mesh.volumes).sum(axis=1)
    #adjEnergy += (rhoUref*rhoUa.getInternalField()**2*mesh.volumes).sum(axis=1)
    #adjEnergy += (rhoEref*rhoEa.getInternalField()**2*mesh.volumes).sum(axis=1)
    # not divided by volumes
    adjEnergy = (rhoref*rhoa.getInternalField()**2/mesh.volumes).sum(axis=1)
    adjEnergy += (rhoUref*rhoUa.getInternalField()**2/mesh.volumes).sum(axis=1)
    adjEnergy += (rhoEref*rhoEa.getInternalField()**2/mesh.volumes).sum(axis=1)
    adjEnergy = (parallel.sum(adjEnergy)**0.5)/(solver.Jref*solver.tref)
    return adjEnergy

def getSymmetrizedAdjointEnergy(solver, rhoa, rhoUa, rhoEa, rho, rhoU, rhoE):
    mesh = solver.mesh
    # optimize this function
    U = rhoU/rho
    g = solver.gamma
    u1, u2, u3 = U.T
    q2 = (u1*u1+u2*u2+u3*u3).reshape(-1,1)
    p = (rhoE-rho*q2/2)*(g-1)
    q2, rho, p, rhoE = q2.flatten(), rho.flatten(), p.flatten(), rhoE.flatten()
    H = g*p/(rho*(g-1)) + q2/2
    #one = np.zeros(mesh.nInternalCells)
    A = np.array([[rho, rho*u1, rho*u2, rho*u3, rhoE],
                  [rho*u1, rho*u1*u1 + p, rho*u1*u2, rho*u1*u3, rho*H*u1],
                  [rho*u2, rho*u2*u1, rho*u2*u2 + p, rho*u2*u3, rho*H*u2],
                  [rho*u3, rho*u3*u1, rho*u3*u2, rho*u3*u3 + p, rho*H*u3],
                  [rhoE, rho*H*u1, rho*H*u2, rho*H*u3, rho*H*H-g*p*p/(rho*(g-1))]]
                  ).transpose(2, 0, 1)
    #A = np.ones((mesh.nInternalCells, 5, 5))
    ## already divided by volumes
    ## not divided by volumes
    rhoa = rhoa/mesh.volumes
    rhoUa = rhoUa/mesh.volumes
    rhoEa = rhoEa/mesh.volumes

    w = np.concatenate((rhoa, rhoUa, rhoEa), axis=1)
    l2norm = np.matmul(w.reshape(-1,1,5), np.matmul(A, w.reshape(-1,5, 1))).reshape(-1,1)
    adjEnergy = parallel.sum(l2norm*mesh.volumes)**0.5
    return adjEnergy

def computeSymmetrizedAdjointEnergy(solver, rhoa, rhoUa, rhoEa, rho, rhoU, rhoE):
    # optimize this function
    mesh = solver.mesh.symMesh
    def computeEnergy(rhoa, rhoUa, rhoEa, rho, rhoU, rhoE, volumes):
        U = rhoU/rho
        u1, u2, u3 = U[0], U[1], U[2]
        q2 = (u1*u1+u2*u2+u3*u3)
        g = solver.gamma
        p = (rhoE - rho*q2/2)*(g-1)
        H = g*p/(rho*(g-1)) + q2/2
        A = Tensor((5,5), [rho, rho*u1, rho*u2, rho*u3, rhoE,
                      rho*u1, rho*u1*u1 + p, rho*u1*u2, rho*u1*u3, rho*H*u1,
                      rho*u2, rho*u2*u1, rho*u2*u2 + p, rho*u2*u3, rho*H*u2,
                      rho*u3, rho*u3*u1, rho*u3*u2, rho*u3*u3 + p, rho*H*u3,
                      rhoE, rho*H*u1, rho*H*u2, rho*H*u3, rho*H*H-g*p*p/(rho*(g-1))])
        # already divided by volumes
        # not divided by volumes
        rhoa = rhoa/volumes
        rhoUa = rhoUa/volumes
        rhoEa = rhoEa/volumes
        w = Tensor((5,), [rhoa, rhoUa[0], rhoUa[1], rhoUa[2], rhoEa])

        l2norm = w.dot(A.tensordot(w))
        return (l2norm*volumes).sum()

    adjEnergy = Zeros((1, 1))
    adjEnergy = Kernel(computeEnergy)(mesh.nInternalCells, (adjEnergy,))(rhoa, rhoUa, rhoEa, rho, rhoU, rhoE, mesh.volumes)
    (adjEnergy,) = ExternalFunctionOp('mpi_allreduce', (adjEnergy,), (Zeros((1,1)),)).outputs
    return adjEnergy

def getAdjointMatrixNorm(rhoa, rhoUa, rhoEa, rho, rhoU, rhoE, U, T, p, *outputs, **kwargs):
    mesh = rho.mesh
    solver = rho.solver
    
    energy = None
    diss = None

    def getArg(arg, default):
        if arg in kwargs:
            if kwargs[arg]:
                return kwargs[arg]
        return default
            
    visc = getArg('visc', 'entropy')
    suffix = '_' + visc
    #suffix = '_' + visc + '_max_2'
    #suffix = '_' + visc + '_min'
    scale = getArg('scale', lambda x: x)
    report = getArg('report', 1)
    if 'scale' in kwargs:
        suffix += '_factor'

    Uref, Tref, pref = solver.Uref, solver.Tref, solver.pref
    gradT, gradrho, gradU, gradp, gradc, divU = outputs
    U = U.getInternal()
    T = T.getInternal()
    p = p.getInternal()
    rho, rhoU, rhoE = solver.conservative(U, T, p)
    rho = rho.field
    rhoU = rhoU.field
    rhoE = rhoE.field
    p = p.field
    U = U.field
    U1 = U[:,[0]]
    U2 = U[:,[1]]
    U3 = U[:,[2]]

    g = solver.gamma
    sg = np.sqrt(g)
    g1 = g-1
    sg1 = np.sqrt(g1)
    sge = sg1*sg
    c = np.sqrt(g*p/rho)
    b = c/sg
    a = sg1*c/sg
    gradb = gradc/sg
    grada = gradc*sg1/sg
    Z = np.zeros_like(divU)
    c2 = c*c
    Us = (U*U).sum(axis=1,keepdims=1)
    B = None
    
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
        
        Ti = np.stack((
                np.hstack((rho/b, Z, Z, Z, Z)),
                np.hstack((rho*U1/b, rho, Z, Z, Z)),
                np.hstack((rho*U2/b, Z, rho, Z, Z)),
                np.hstack((rho*U3/b, Z, Z, rho, Z)),
                np.hstack((rho*(2*c2/(g1*g)+Us)/(2*b), rho*U1, rho*U2, rho*U3, c*rho/sge)),
            ), axis=2)
        M = M1/2-M2

    # Entropy
    elif visc == 'entropy_hughes':
        rho = rho[:,0]
        u1 = U[:,0]
        u2 = U[:,1]
        u3 = U[:,2]
        p = p[:,0]
        q2 = (u1*u1+u2*u2+u3*u3)
        H = g*p/(rho*(g-1)) + q2/2
        rE = p/(g-1) + 0.5*rho*q2
        zero = np.zeros_like(rho)

        #rhou1 = rhoU[:,0]
        #rhou2 = rhoU[:,1]
        #rhou3 = rhoU[:,2]
        #rhoE = rhoE[:,0]
        #q2 = u1*u1 + u2*u2 + u3*u3
        #from .symmetrizations.entropy_barth_numerical import expression
        #one = np.ones_like(rho)
        #A0U, AU, A0, A = expression(rho,rhou1,rhou2,rhou3,rhoE, g*one, one, zero)
        #A0U = np.array(A0U).transpose(1, 0).reshape((-1, 5, 5, 5))
        #AU = np.array(AU).transpose(1, 0).reshape((-1, 5, 3, 5, 5))
        #A = np.array(A).transpose(1, 0).reshape((-1, 5, 3, 5))
        #A0 = np.array(A0).transpose(1, 0).reshape((-1, 5, 5))
        #Twq = np.array([[one,zero,zero,zero,zero],
        #                [u1,rho,zero,zero,zero],
        #                [u2,zero,rho,zero,zero],
        #                [u3,zero,zero,rho,zero],
        #                [q2/2,rho*u1,rho*u2,rho*u3,one/g1]]).transpose((2, 0, 1))
        #G = np.concatenate((gradrho.reshape(-1,1,3), gradU, gradp.reshape(-1,1,3)), axis=1)
        #Gw = np.matmul(Twq, G)
        #AtU = np.einsum('pijkm,pkl->pijlm', AU, A0) + np.einsum('pijk,pklm->pijlm', A, A0U)
        #M1 = np.einsum('pkjil,plj->pki', AtU, Gw)
        #M2 = np.einsum('pijk,pklm,pml->pij', A0U, A, Gw)
        #M = -M1 + M2

        #suffix += '_test'
        #gradp = 0*gradp
        #gradrho = 0*gradrho
        #gradU = 0*gradU
        from .symmetrizations.entropy_barth_mathematica import expression
        M = expression(rho,u1,u2,u3,p,gradrho,gradU[:,0,:],gradU[:,1,:],gradU[:,2,:],gradp,g,zero)
        M = np.array(M).transpose((2, 0, 1))
        # non dimensionalizer
        #X = np.diag([1, 1./Uref, 1./Uref, 1./Uref, 1/pref])
        #M = np.matmul(np.matmul(X, M), X)
        # A0 weighted norm
        B = np.array([[rho, rho*u1, rho*u2, rho*u3, rE],
                  [rho*u1, rho*u1*u1 + p, rho*u1*u2, rho*u1*u3, rho*H*u1],
                  [rho*u2, rho*u2*u1, rho*u2*u2 + p, rho*u2*u3, rho*H*u2],
                  [rho*u3, rho*u3*u1, rho*u3*u2, rho*u3*u3 + p, rho*H*u3],
                  [rE, rho*H*u1, rho*H*u2, rho*H*u3, rho*H*H-g*p*p/(rho*(g-1))]]
                  ).transpose(2, 0, 1)

    #elif visc == 'entropy_hughes':
    #    #pref = 1.
    #    rho = rho[:,0]
    #    p = p[:,0]
    #    u1 = U[:,0]
    #    u2 = U[:,1]
    #    u3 = U[:,2]
    #    re = p/g1
    #    q2 = u1*u1 + u2*u2 + u3*u3
    #    rE = re + rho*q2/2
    #    s = np.log(g1*re/pref/rho**g)
    #    V1 = (-rE + re*(g+1-s))/re
    #    V2 = rho*u1/re
    #    V3 = rho*u2/re
    #    V4 = rho*u3/re
    #    V5 = -rho/re

    #    one = np.ones_like(rho)
    #    zero = np.zeros_like(rho)
    #    Twq = np.array([[one,zero,zero,zero,zero],
    #                    [u1,rho,zero,zero,zero],
    #                    [u2,zero,rho,zero,zero],
    #                    [u3,zero,zero,rho,zero],
    #                    [q2/2,rho*u1,rho*u2,rho*u3,one/g1]]).transpose((2, 0, 1))
    #    from .symmetrizations.entropy_hughes_numerical import expression
    #    #Twq, B, Y, A0I, A = expression(rho,u1,u2,u3,p,V1,V2,V3,V4,V5, pref, g, re)
    #    B, Y, A0I, A = expression(rho,u1,u2,u3,p,V1,V2,V3,V4,V5, pref, g)
    #    Tvq = np.matmul(A0I, Twq)
    #    G = np.concatenate((gradrho.reshape(-1,1,3), gradU, gradp.reshape(-1,1,3)), axis=1)
    #    Gv = np.matmul(Tvq, G)
    #    M2 = np.einsum('pikj,pjl,plmn,pnm->pik', Y, A0I, A, Gv)
    #    M1 = np.einsum('pkjil,plj->pki', B, Gv)
    #    M = -M1 + M2
    #    #test = IOField('test' + suffix, Gv[:,1,:], (3,), boundary=mesh.calculatedBoundary)
    #    #test.write()

    elif visc == 'turkel' or visc == 'uniform':
        M1 = np.stack((np.hstack((divU, gradc, Z)),
                   np.hstack((gradc[:,[0]], divU, Z, Z, Z)),
                   np.hstack((gradc[:,[1]], Z, divU, Z, Z)),
                   np.hstack((gradc[:,[2]], Z, Z, divU, Z)),
                   np.hstack((Z, Z, Z, Z, divU))),
                   axis=1)

        M2 = np.concatenate((np.hstack((g1*divU/2, gradp/(rho*c), divU*pref/(2*rho*c*Uref))).reshape(-1,1,5),
                        np.dstack(((g1*gradp/(2*rho*c)).reshape(-1,3,1), gradU, (gradp*pref/(2*g*p*rho*Uref)).reshape((-1, 3, 1)))),
                        np.hstack((Z, (gradp-c*c*gradrho)*Uref/pref, Z)).reshape(-1,1,5)),
                        axis=1)
        M = M1/2-M2

        Ti = np.stack((
            np.hstack((rho*Uref/c, Z, Z, Z, -pref/c2)),
            np.hstack((rho*U1*Uref/c, rho*Uref, Z, Z, -pref*U1/c2)),                                    #suffix += '_factor'
            np.hstack((rho*U2*Uref/c, Z, rho*Uref, Z, -pref*U2/c2)),                                    #suffix += '_factor'
            np.hstack((rho*U3*Uref/c, Z, Z, rho*Uref, -pref*U3/c2)),                                    #suffix += '_factor'
            np.hstack((c*rho*Uref/g1 + rho*Us*Uref/(2*c), rho*U1*Uref, rho*U2*Uref, rho*U3*Uref, -pref*Us/(2*c2))),    #MS = (M + M.transpose((0, 2, 1)))/2
        ), axis=2)                                                                                   #M_2norm = np.linalg.eigvalsh(MS)[:,[-1]]
    else:
        raise Exception('factor not recognized')
    #M_2norm = IOField('M_2norm_old' + suffix, M_2norm, (1,), boundary=mesh.calculatedBoundary)
    #M_2norm.write()
    
    def dot(a, b):
        return np.sum(a*b.reshape(-1,1,5), axis=-1)

    Mc = M
    MS = (Mc + Mc.transpose((0, 2, 1)))/2
    # max eigenvalue
    if B is None:
        M_2norm = np.linalg.eigvalsh(MS)[:,[-1]]
    else:
        eig = np.linalg.eigvals(np.matmul(np.linalg.inv(B),MS))
        M_2norm = np.sort(eig,axis=1)[:,[-1]]
    # second max eigenvalue
    #M_2norm = np.linalg.eigvalsh(MS)[:,[-2]]
    # min eigenvalue
    #M_2norm = np.linalg.eigvalsh(MS)[:,[0]]

    M_2norm = scale(M_2norm)

    def inner(F, G):
        if not hasattr(getAdjointMatrixNorm, 'Vs'):
            getAdjointMatrixNorm.Vs = parallel.sum(mesh.volumes)
        Vs = getAdjointMatrixNorm.Vs
        return parallel.sum(F*G*mesh.volumes)/Vs
    l2_norm = lambda F: np.sqrt(inner(F, F))

    if visc == "uniform":
        M_2norm = np.ones_like(M_2norm)
    if report:
        getAdjointMatrixNorm.l2_norm = l2_norm(M_2norm)
        pprint(getAdjointMatrixNorm.l2_norm)
    linf_norm = parallel.max(M_2norm)
    M_2norm /= getAdjointMatrixNorm.l2_norm
    pprint(linf_norm/getAdjointMatrixNorm.l2_norm)
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

        corr = inner(diss.field, energy.field)/(l2_norm(diss.field)*l2_norm(energy.field))
        parallel.pprint('energy and M_2norm corr:', corr)

        diss.defaultComplete()
        energy.defaultComplete()

    M_2norm.defaultComplete()

    return M_2norm, energy, diss


def getAdjointViscosity(rho, rhoU, rhoE, scaling, outputs=None, init=True, **kwargs):
    solver = rho.solver
    mesh = rho.mesh
    start = time.time()
    if init:
        rho, rhoU, rhoE = solver.initFields((rho, rhoU, rhoE))
    U, T, p = solver.primitive(rho, rhoU, rhoE)
    if not outputs:
        outputs = computeGradients(solver, U, T, p)
    M_2norm = getAdjointMatrixNorm(None, None, None, rho, rhoU, rhoE, U, T, p, *outputs, **kwargs)[0]
    viscosity = M_2norm*float(scaling)
    viscosity.name = 'mua'
    viscosity.boundary = mesh.defaultBoundary
    return viscosity

def computeAdjointViscosity(solver, viscosityType, rho, rhoU, rhoE, scaling):
    g = solver.gamma
    mesh = solver.mesh.symMesh

    def _meshArgs(start=0):
        return [x[start] for x in mesh.getTensor()]

    def gradients(U, T, p, *mesh, **options):
        mesh = Mesh.container(mesh)
        neighbour = options.pop('neighbour', True)
        boundary = options.pop('boundary', False)

        if boundary:
            UF = U.extract(mesh.neighbour)
            pF = p.extract(mesh.neighbour)
            TF = T.extract(mesh.neighbour)
            cF = (g*TF*solver.R).sqrt()
        else:
            UF = interp.central(U, mesh)
            pF = interp.central(p, mesh)
            TLF, TRF = T.extract(mesh.owner), T.extract(mesh.neighbour)
            cLF, cRF = (g*TLF*solver.R).sqrt(), (g*TRF*solver.R).sqrt()
            cF = cRF*(1-mesh.weights) + cLF*mesh.weights

        gradU = op.grad(UF, mesh, neighbour)
        divU = op.div(UF.dot(mesh.normals), mesh, neighbour)
        gradp = op.grad(pF, mesh, neighbour)
        gradc = op.grad(cF, mesh, neighbour)
        return gradU, divU, gradp, gradc
    computeGradients = Kernel(gradients)
    boundaryComputeGradients = Kernel(gradients)
    coupledComputeGradients = Kernel(gradients)

    U, T, p = Zeros((mesh.nCells, 3)), Zeros((mesh.nCells, 1)), Zeros((mesh.nCells, 1))
    outputs = solver._primitive(mesh.nInternalCells, (U, T, p))(rho, rhoU, rhoE)
    # boundary update
    outputs = solver.boundaryInit(*outputs)
    outputs = solver.boundary(*outputs)
    outputs = solver.boundaryEnd(*outputs)
    U, T, p = outputs

    meshArgs = _meshArgs()
    gradU, divU, gradp, gradc = Zeros((mesh.nInternalCells, 3, 3)), Zeros((mesh.nInternalCells, 1)), Zeros((mesh.nInternalCells, 3)), Zeros((mesh.nInternalCells, 3))
    outputs = computeGradients(mesh.nInternalFaces, (gradU, divU, gradp, gradc))(U, T, p, *meshArgs)
    for patchID in solver.mesh.sortedPatches:
        startFace, nFaces = mesh.boundary[patchID]['startFace'], mesh.boundary[patchID]['nFaces']
        patchType = solver.mesh.boundary[patchID]['type']
        meshArgs = _meshArgs(startFace)
        if patchType in config.coupledPatches:
            outputs = coupledComputeGradients(nFaces, outputs)(U, T, p, neighbour=False, boundary=False, *meshArgs)
            outputs[0].args[0].info += [[x.shape for x in solver.mesh.getTensor()], patchID, solver.mesh.boundary[patchID]['startFace'], solver.mesh.boundary[patchID]['nFaces']]
        else:
            outputs = boundaryComputeGradients(nFaces, outputs)(U, T, p, neighbour=False, boundary=True, *meshArgs)
    meshArgs = _meshArgs(mesh.nLocalFaces)
    outputs = coupledComputeGradients(mesh.nRemoteCells, outputs)(U, T, p, neighbour=False, boundary=False, *meshArgs)
    gradU, divU, gradp, gradc = outputs

    def getMaxEigenvalue(U, T, p, gradU, divU, gradp, gradc):
        Uref, Tref, pref = solver.Uref, solver.Tref, solver.pref
        sg = np.sqrt(g)
        g1 = g-1
        sg1 = np.sqrt(g1)
        sge = sg1*sg

        rho, _, _ = solver.conservative(U, T, p)
        c = (g*p/rho).sqrt()
        gradrho = g*(gradp-c*p)/(c*c)
        b = c/sg
        a = sg1*c/sg
        gradb = gradc/sg
        grada = gradc*sg1/sg
        Z = Tensor((1,), [ConstantOp(0.)])
        U1, U2, U3 = U[0], U[1], U[2]
        Us = U.magSqr()
        c2 = c*c
        B = Tensor((5,5), [rho, rho, rho, rho, rho,
                           rho, rho, rho, rho, rho,
                           rho, rho, rho, rho, rho,
                           rho, rho, rho, rho, rho,
                           rho, rho, rho, rho, rho])

        if viscosityType == 'abarbanel' or viscosityType == 'uniform':
            M1 = Tensor((5, 5), [divU, gradb[0], gradb[1], gradb[2], Z,
                                 gradb[0], divU, Z, Z, grada[0],
                                 gradb[1], Z, divU, Z, grada[1],
                                 gradb[2], Z, Z, divU, grada[2],
                                 Z, grada[0], grada[1], grada[2], divU])
            tmp1 = b*gradrho/rho
            tmp2 = a*gradp/(2*p)
            tmp3 = 2*grada/g1

            M2 = Tensor((5, 5), [Z, tmp1[0], tmp1[1], tmp1[2], sg1*divU/2,
                                 Z, gradU[0,0], gradU[0,1], gradU[0,2], tmp2[0],
                                 Z, gradU[1,0], gradU[1,1], gradU[1,2], tmp2[1],
                                 Z, gradU[2,0], gradU[2,1], gradU[2,2], tmp2[2],
                                 Z, tmp3[0], tmp3[1], tmp3[2], g1*divU/2])
            M = M1/2-M2
        elif viscosityType == 'turkel':
            M1 = Tensor((5, 5), [divU, gradc[0], gradc[1], gradc[2], Z,
                                 gradc[0], divU, Z, Z, Z,
                                 gradc[1], Z, divU, Z, Z,
                                 gradc[2], Z, Z, divU, Z,
                                 Z, Z, Z, Z, divU])
            tmp1 = gradp/(rho*c)
            tmp2 = g1*gradp/(2*rho*c)
            tmp3 = gradp*pref/(2*g*p*rho*Uref)
            tmp4 = (gradp-c*c*gradrho)*Uref/pref

            M2 = Tensor((5, 5), [g1*divU/2, tmp1[0], tmp1[1], tmp1[2], divU*pref/(2*rho*c*Uref),
                                 tmp2[0], gradU[0,0], gradU[0,1], gradU[0,2], tmp3[0],
                                 tmp2[1], gradU[1,0], gradU[1,1], gradU[1,2], tmp3[1],
                                 tmp2[2], gradU[2,0], gradU[2,1], gradU[2,2], tmp3[2],
                                 Z, tmp4[0], tmp4[1], tmp4[2], g1*divU/2])
            M = M1/2-M2
        #elif viscosityType == 'entropy_hughes':
        #    from .symmetrizations.entropy_hughes_gen_code import expression
        #    M1, M2 = expression(rho, U, p, gradrho, gradU, gradp, g, Z)
        #    M1 = Tensor((5,5), M1)
        #    M2 = Tensor((5,5), M2)
        #    M = -(M1 + M2)

        elif viscosityType == 'entropy_hughes':
            from .symmetrizations.entropy_barth_mathematica import expression_code
            M = expression_code(rho, U, p, gradrho, gradU, gradp, g, Z)

            M = [item for sublist in M for item in sublist]
            #X = [1., 1./Uref, 1./Uref, 1./Uref, 1/pref]
            #M = [item*X[i]*X[j] for i, sublist in enumerate(M) for j, item in enumerate(sublist)]
            M = Tensor((5,5), M)

            u1, u2, u3 = U[0], U[1], U[2]
            q2 = (u1*u1+u2*u2+u3*u3)
            H = g*p/(rho*(g-1)) + q2/2
            rE = p/(g-1) + 0.5*rho*q2
            B = Tensor((5,5), [rho, rho*u1, rho*u2, rho*u3, rE,
                          rho*u1, rho*u1*u1 + p, rho*u1*u2, rho*u1*u3, rho*H*u1,
                          rho*u2, rho*u2*u1, rho*u2*u2 + p, rho*u2*u3, rho*H*u2,
                          rho*u3, rho*u3*u1, rho*u3*u2, rho*u3*u3 + p, rho*H*u3,
                          rE, rho*H*u1, rho*H*u2, rho*H*u3, rho*H*H-g*p*p/(rho*(g-1))])

        else:
            raise Exception('symmetrizer not recognized')

        
        #Ti = Tensor((5, 5), [rho/b, Z, Z, Z, Z,
        #                     rho*U1/b, rho, Z, Z, Z,
        #                     rho*U2/b, Z, rho, Z, Z,
        #                     rho*U3/b, Z, Z, rho, Z,
        #                     rho*(c2*2/(g1*g)+Us)/(2*b), rho*U1, rho*U2, rho*U3, c*rho/sge])
        #Ti = Ti.transpose()
        #X = np.diag([1, 1./Uref, 1./Uref, 1./Uref, 1/pref])
        #TiX = Ti.matmul(X)

        #Mc = TiX.transpose().matmul(M.matmul(TiX))
        Mc = M
        MS = (Mc + Mc.transpose())/2
        return MS, B

    def constant(M_2norm):
        return M_2norm + 1

    M_2norm = Zeros((mesh.nInternalCells, 1))
    if viscosityType == 'uniform':
        M_2norm = Kernel(constant)(mesh.nInternalCells)(M_2norm)
    else:
        MS = Zeros((mesh.nInternalCells, 5, 5))
        B = Zeros((mesh.nInternalCells, 5, 5))
        MS, B = Kernel(getMaxEigenvalue)(mesh.nInternalCells, (MS, B))(U, T, p, gradU, divU, gradp, gradc)
        if viscosityType != 'entropy_hughes':
            (M_2norm,) = ExternalFunctionOp('get_max_eigenvalue', (MS,), (M_2norm,)).outputs
        else:
            (M_2norm,) = ExternalFunctionOp('get_max_generalized_eigenvalue', (MS, B), (M_2norm,)).outputs

    def computeVolume(volumes):
        return volumes.sum()
    V = Zeros((1,1))
    V = Kernel(computeVolume)(mesh.nInternalCells, (V,))(mesh.volumes)
    (V,) = ExternalFunctionOp('mpi_allreduce', (V,), (Zeros((1,1)),)).outputs

    def computeNorm(M_2norm, volumes, V):
        V = V.scalar()
        N = (M_2norm*M_2norm*volumes/V).sum()
        return N
    N = Zeros((1,1))
    N = Kernel(computeNorm)(mesh.nInternalCells, (N,))(M_2norm, mesh.volumes, V)
    (N,) = ExternalFunctionOp('mpi_allreduce', (N,), (Zeros((1,1)),)).outputs

    def scaleM(M_2norm, N, scaling):
        N, scaling = N.scalar(), scaling.scalar()
        return M_2norm*scaling/N.sqrt()
    M_2norm_out = Zeros((mesh.nCells, 1))
    M_2norm = Kernel(scaleM)(mesh.nInternalCells, (M_2norm_out,))(M_2norm, N, scaling)

    (phi,) = solver.boundaryInit(M_2norm)
    phi = CellField('M_2norm', None, (1,)).updateGhostCells(phi)
    (phi,) = ExternalFunctionOp('mpi', (phi,), (phi,)).outputs
    (phi,) = solver.boundaryEnd(phi)
    M_2norm = phi

    def interpolate(M_2norm, *mesh):
        mesh = Mesh.container(mesh)
        M_2norm = interp.central(M_2norm, mesh)
        return M_2norm
    meshArgs = _meshArgs()
    DT = Zeros((mesh.nFaces, 1))
    DT = Kernel(interpolate)(mesh.nFaces, (DT,))(M_2norm, *meshArgs)
    return M_2norm, DT

def viscositySolver(solver, rho, rhoU, rhoE, rhoa, rhoUa, rhoEa, DT):
    mesh = solver.mesh.symMesh
    def divideFields(rhoa, rhoUa, rhoEa, volumes):
        return rhoa/volumes, rhoUa[0]/volumes, rhoUa[1]/volumes, rhoUa[2]/volumes, rhoEa/volumes
    fields = Kernel(divideFields)(mesh.nInternalCells)(rhoa, rhoUa, rhoEa, mesh.volumes)

    def getFaceData(DT, areas, deltas):
        return areas*DT/deltas
    DTF = Kernel(getFaceData)(mesh.nFaces)(DT, mesh.areas, mesh.deltas)

    inputs = fields + (rho, rhoU, rhoE, DTF, solver.dt)
    outputs = tuple([Zeros(x.shape) for x in fields])
    fields = ExternalFunctionOp('apply_adjoint_viscosity', inputs, outputs).outputs

    def multiplyFields(phi1, phi2, phi3, phi4, phi5, volumes):
        rhoa = phi1*volumes
        rhoUa = Tensor((3,), [phi2*volumes, phi3*volumes, phi4*volumes])
        rhoEa = phi5*volumes
        return rhoa, rhoUa, rhoEa
    return Kernel(multiplyFields)(mesh.nInternalCells)(*(fields + (mesh.volumes,)))

