#!/usr/bin/python2
import numpy as np
import sys
import time

import config
from config import ad
from parallel import pprint

from field import Field, CellField, IOField
from op import  div, snGrad, grad, laplacian, internal_sum
from solver import Solver
from interp import central, TVD_dual
import riemann

logger = config.Logger(__name__)

class RCF(Solver):
    defaultConfig = Solver.defaultConfig.copy()
    defaultConfig.update({
                             'R': 8.314, 
                             'Cp': 1004.5, 
                             'gamma': 1.4, 
                             'mu': lambda T:  1.4792e-06*T**1.5/(T+116.), 
                             'Pr': 0.7, 
                             'CFL': 0.6,
                             'stepFactor': 1.2,
                             'timeIntegrator': 'SSPRK', 'nStages': 3,
                             'riemannSolver': 'eulerRoe',
                             'source': lambda x: [0, 0, 0]
                        })

    def __init__(self, case, **userConfig):
        super(RCF, self).__init__(case, **userConfig)

        self.Cv = self.Cp/self.gamma
        self.kappa = lambda mu, T: mu*(self.Cp/self.Pr)
        self.names = ['rho', 'rhoU', 'rhoE']
        self.dimensions = [(1,), (3,), (1,)]
        self.riemannSolver = getattr(riemann, self.riemannSolver)

    def primitive(self, rho, rhoU, rhoE):
        logger.info('converting fields to primitive')
        U = rhoU/rho
        E = rhoE/rho
        e = E - 0.5*U.magSqr()
        p = (self.gamma-1)*rho*e
        T = e*(1./self.Cv)
        U.name, T.name, p.name = 'U', 'T', 'p'
        return U, T, p

    def conservative(self, U, T, p):
        logger.info('converting fields to conservative')
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        E = e + 0.5*U.magSqr()
        rhoU = U*rho
        rhoE = rho*E
        rho.name, rhoU.name, rhoE.name = self.names
        return rho, rhoU, rhoE

    def getRho(self, T, p):
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        return rho

    def getBCFields(self):
        return self.U.phi, self.T.phi, self.p.phi

    def initFields(self, t):
        # IO Fields, with phi attribute as a CellField
        self.U = IOField.read('U', self.mesh, t)
        self.T = IOField.read('T', self.mesh, t)
        self.p = IOField.read('p', self.mesh, t)
        UI = self.U.complete()
        TI = self.T.complete()
        pI = self.p.complete()
        UN, TN, pN = self.U.phi.field, self.T.phi.field, self.p.phi.field 
        if not hasattr(self, 'initFunc'):
            self.initFunc = self.function([UI, TI, pI], [UN, TN, pN], 'init')
        self.U.field, self.T.field, self.p.field = self.initFunc(self.U.field, self.T.field, self.p.field)
        return self.conservative(self.U, self.T, self.p)
    
    def writeFields(self, fields, t):
        for phi in fields:
            phi.write(t)
        U, T, p = self.primitive(*fields)
        self.U.field, self.T.field, self.p.field = U.field, T.field, p.field
        self.U.write(t)
        self.T.write(t)
        self.p.write(t)
           
    def equation(self, rhoP, rhoUP, rhoEP, exit=False):
        logger.info('computing RHS/LHS')
        mesh = self.mesh
        paddedMesh = mesh.paddedMesh

        # phi is in paddedMesh form, needs to be copied to regular
        # phi from phiPaddedMesh
        UP, TP, pP = self.primitive(rhoP, rhoUP, rhoEP)
        U = CellField.getOrigField(UP)
        T = CellField.getOrigField(TP)
        p = CellField.getOrigField(pP)

        self.local = mesh.nCells
        self.remote = mesh.nCells
        
        # gradient evaluated using gauss integration rule
        #rho = CellField.getOrigField(rhoP)
        #rhoU = CellField.getOrigField(rhoUP)
        #rhoE = CellField.getOrigField(rhoEP)
        #gradRho = grad(central(rhoP, paddedMesh), ghost=True)
        #gradRhoU = grad(central(rhoUP, paddedMesh), ghost=True)
        #gradRhoE = grad(central(rhoEP, paddedMesh), ghost=True)

        ## face reconstruction
        #rhoLF, rhoRF = TVD_dual(rho, gradRho)
        #rhoULF, rhoURF = TVD_dual(rhoU, gradRhoU)
        #rhoELF, rhoERF = TVD_dual(rhoE, gradRhoE)
        #ULF, TLF, pLF = self.primitive(rhoLF, rhoULF, rhoELF)
        #URF, TRF, pRF = self.primitive(rhoRF, rhoURF, rhoERF)

        # gradient evaluated using gauss integration rule
        gradU = grad(central(UP, paddedMesh), ghost=True)
        gradT = grad(central(TP, paddedMesh), ghost=True)
        gradp = grad(central(pP, paddedMesh), ghost=True)
        # for zeroGradient boundary
        UB, TB, pB = self.getBCFields()
        UB.grad, TB.grad, pB.grad = gradU, gradT, gradp

        # face reconstruction
        ULF, URF = TVD_dual(U, gradU)
        TLF, TRF = TVD_dual(T, gradT)
        pLF, pRF = TVD_dual(p, gradp)
        rhoLF, rhoULF, rhoELF = self.conservative(ULF, TLF, pLF)
        rhoRF, rhoURF, rhoERF = self.conservative(URF, TRF, pRF)

        rhoFlux, rhoUFlux, rhoEFlux, aF, UnF = self.riemannSolver(mesh, self.gamma, \
                pLF, pRF, TLF, TRF, ULF, URF, \
                rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF)

        # viscous part
        TF = 0.5*(TLF + TRF)
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)
        
        gradUTF = central(gradU.transpose(), mesh)
        sigmaF = (snGrad(U) + gradUTF.dotN() - (2./3)*mesh.Normals*gradUTF.trace())*mu
        UF = 0.5*(ULF + URF)
        sigmadotUF = sigmaF.dot(UF)

        # source terms
        source = self.source(self)

        # CFL based time step
        self.aF = aF
        aF = UnF.abs() + aF
        self.dtc = 2*self.CFL/internal_sum(aF, mesh, absolute=True)
        
        return [div(rhoFlux) - source[0], \
                #ddt(rhoU, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF) - source[1],
                div(rhoUFlux) - div(sigmaF) - source[1], \
                div(rhoEFlux) - laplacian(T, kappa) + div(sigmadotUF) - source[2]]

    def boundary(self, rhoN, rhoUN, rhoEN):
        logger.info('correcting boundary')
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        U, T, p = self.getBCFields()
        U.setInternalField(UN.field, reset=True)
        T.setInternalField(TN.field, reset=True)
        p.setInternalField(pN.field, reset=True)
        return self.conservative(U, T, p)
    
if __name__ == "__main__":
    if len(sys.argv) > 2:
        case = sys.argv[1]
        time = float(sys.argv[2])
    else:
        pprint('WTF')
        exit()

    solver = RCF(case, timeIntegrator='euler', CFL=0.5)
    solver.run(startTime=time, dt=1e-9, nSteps=20000, writeInterval=500)
    #solver = RCF(case, timeIntegrator='SSPRK', CFL=0.7, Cp=2.5, mu=lambda T: config.VSMALL*T)
    #solver.run(startTime=time, dt=1e-4, nSteps=60000, writeInterval=100)
