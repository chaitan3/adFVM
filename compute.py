#!/usr/bin/python2
import config, parallel
from config import ad
from parallel import pprint

import numpy as np

from pyRCF import RCF
from field import IOField, CellField
from op import div, grad
from interp import central

def computeFields(solver):
    mesh = solver.mesh
    g = solver.gamma
    SF = ad.matrix()
    p, U, T = solver.unstackFields(SF, CellField)
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

    computer = solver.function([SF], [gradrho.field, gradU.field, gradp.field, gradc.field, divU.field], 'compute')
    return computer

def getRhoaByV(rhoa):
    mesh = rhoa.mesh
    rhoaByV = np.zeros((mesh.origMesh.nCells, 1))
    nInternalCells = mesh.origMesh.nInternalCells
    rhoaByV[:nInternalCells] = rhoa.field[:nInternalCells]/mesh.origMesh.volumes
    rhoaByV = IOField('rhoaByV', rhoaByV, (1,), boundary=mesh.calculatedBoundary)
    return rhoaByV

def getAdjointEnergy(rhoa, rhoUa, rhoEa, solver)
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
    #print np.hstack((divU, gradb, Z)).shape,np.hstack((gradb[:,[0]], divU, Z, Z, grada[:,[0]])).shape,np.hstack((Z, grada, divU)).shape
    M1 = np.dstack((np.hstack((divU, gradb, Z)),
               np.hstack((gradb[:,[0]], divU, Z, Z, grada[:,[0]])),
               np.hstack((gradb[:,[1]], Z, divU, Z, grada[:,[1]])),
               np.hstack((gradb[:,[2]], Z, Z, divU, grada[:,[2]])),
               np.hstack((Z, grada, divU))))

    M2 = np.dstack((np.hstack((Z, b*gradrho/rho, sg1*divU/2)),
                    np.hstack((np.dstack((Z,Z,Z)), gradU, (a*gradp/(2*p)).reshape(-1, 1, 3))),
                    np.hstack((Z, 2*grada/g1, g1*divU/2))))
    M = M1-M2
    U, S, V = np.linalg.svd(M)
    #V = np.ascontiguousarray(V.transpose((0, 2, 1)))
    #A = (((U*V).sum(axis=1)*S) > 0.)
    #A = A*1.
    #Smax = (A*S).max(axis=1, keepdims=True)
    #Imax = (A*S).argmax(axis=1)
    #idx = np.arange(0, mesh.origMesh.nCells)
    #Umax = U[idx, Imax]*Smax
    #Vmax = V[idx, Imax]*Smax
    ## transform, U, V
    #names = [name + '_A' for name in solver.names]
    #F = solver.unstackFields(A, IOField, names, boundary=mesh.calculatedBoundary)
    #names = [name + '_S' for name in solver.names]
    #F += solver.unstackFields(S, IOField, names, boundary=mesh.calculatedBoundary)
    #for index in range(0, U.shape[2]):
    #    names = [name + '_U' + str(index + 1) for name in solver.names]
    #    F += solver.unstackFields(U[:,:,index], IOField, names, boundary=mesh.calculatedBoundary)
    #    names = [name + '_V' + str(index + 1) for name in solver.names]
    #    F += solver.unstackFields(V[:,:,index], IOField, names, boundary=mesh.calculatedBoundary)
    #Smax = IOField('Smax', Smax, (1,), boundary=mesh.calculatedBoundary)
    #F.append(Smax)
    #names = [name + '_Umax' for name in solver.names]
    #F += solver.unstackFields(Umax, IOField, names, boundary=mesh.calculatedBoundary)
    #names = [name + '_Vmax' for name in solver.names]
    #F += solver.unstackFields(Vmax, IOField, names, boundary=mesh.calculatedBoundary)
    #for phi in F:
    #    phi.field = np.ascontiguousarray(phi.field)
    #return F

    M_2norm = np.ascontiguousarray(S[:, [0]])
    M_2norm = IOField('M_2norm', M_2norm, (1,), boundary=mesh.calculatedBoundary)
    return [M_2norm]
 
if __name__ == "__main__":
    import time as timer
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    parser.add_argument('time', nargs='+', type=float)
    user = parser.parse_args(config.args)

    names = ['gradrho', 'gradU', 'gradp', 'gradc', 'divU']
    dimensions = [(3,), (3,3), (3,),(3,),(1,)]

    solver = RCF(user.case)
    mesh = solver.mesh
    #solver.initialize(user.time[0])
    #computer = computeFields(solver)

    if config.compile:
        exit()

    for index, time in enumerate(user.time):
        pprint('Time:', time)
        start = timer.time()
        #rho, rhoU, rhoE = solver.initFields(time)
        #U, T, p = solver.U, solver.T, solver.p
        #SF = solver.stackFields([p, U, T], np)
        #outputs = computer(SF)
        #for field, name, dim in zip(outputs, names, dimensions):
        #    IO = IOField(name, field, dim)
        #    if len(dim) != 2:
        #        IO.write(time)
        #pprint()

        # rhoaByV
        rhoa = IOField.read('rhoa', mesh, time)
        rhoaByV = getRhoaByV(rhoa)
        rhoaByV.write(time)
        pprint()

        # adjoint energy
        rhoUa = IOField.read('rhoUa', mesh, time)
        rhoEa = IOField.read('rhoEa', mesh, time)
        adjEnergy = getAdjointEnergy(rhoa, rhoUa, rhoEa, solver)
        pprint('L2 norm adjoint', time, adjEnergy)
        pprint()

        # adjoint blowup
        #fields = getAdjointNorm(rho, rhoU, rhoE, U, T, p, *outputs)
        #for phi in fields:
        #    phi.write(time)#, skipProcessor=True)
        #end = timer.time()
        pprint('Time for computing: {0}'.format(end-start))

        pprint()
