#!/usr/bin/python2
import numpy as np
import sys
import time

from field import Field, CellField
from op import  div, snGrad, grad, ddt, laplacian
from solver import Solver
from solver import forget
from interp import interpolate, TVD_dual

from config import ad, Logger
from parallel import pprint
logger = Logger(__name__)
import config, parallel

class RCF(Solver):
    defaultConfig = Solver.defaultConfig.copy()
    defaultConfig.update({
                             'R': 8.314, 
                             'Cp': 1004.5, 
                             'gamma': 1.4, 
                             'mu': lambda T:  1.4792e-06*T**1.5/(T+116), 
                             'Pr': 0.7, 
                             'CFL': 0.6,
                             'stepFactor': 1.2,
                             'timeIntegrator': 'RK', 
                             'source': lambda x: [0, 0, 0]
                        })

    def __init__(self, case, **userConfig):
        super(RCF, self).__init__(case, **userConfig)

        self.Cv = self.Cp/self.gamma
        self.kappa = lambda mu, T: mu*(self.Cp/self.Pr)
        self.names = ['rho', 'rhoU', 'rhoE']
        self.dimensions = [(1,), (3,), (1,)]

    def primitive(self, rho, rhoU, rhoE):
        logger.info('converting fields to primitive')
        U = rhoU/rho
        E = rhoE/rho
        e = E - 0.5*U.magSqr()
        p = (self.gamma-1)*rho*e
        T = e*(1./self.Cv)
        return U, T, p

    def conservative(self, U, T, p):
        logger.info('converting fields to conservative')
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        E = e + 0.5*U.magSqr()
        rhoU = rho*U
        rhoE = rho*E
        rho.name, rhoU.name, rhoE.name = self.names
        return rho, rhoU, rhoE

    def initFields(self, t):
        self.p = CellField.read('p', t)
        self.T = CellField.read('T', t)
        self.U = CellField.read('U', t)
        return self.conservative(self.U, self.T, self.p)
    
    def clearFields(self, fields):
        forget([self.U, self.T, self.p])

    def writeFields(self, fields):
        for phi in fields:
            phi.write(self.t)
        self.U.write(self.t)
        self.T.write(self.t)
        self.p.write(self.t)
           
    def setDt(self, aFbyD):
        logger.info('computing new time step')
        self.dt = min(self.dt*self.stepFactor, self.CFL/parallel.max(aFbyD))
        super(RCF, self).setDt()

    def equation(self, rho, rhoU, rhoE):
        logger.info('computing RHS/LHS')
        mesh = self.mesh

        # interpolation
        rhoLF, rhoRF = TVD_dual(rho)
        rhoULF, rhoURF = TVD_dual(rhoU)
        rhoELF, rhoERF = TVD_dual(rhoE)
        ULF, TLF, pLF = self.primitive(rhoLF, rhoULF, rhoELF)
        URF, TRF, pRF = self.primitive(rhoRF, rhoURF, rhoERF)
        U, T, p = self.primitive(rho, rhoU, rhoE)

        # numerical viscosity
        cLF, cRF = (self.gamma*pLF/rhoLF)**0.5, (self.gamma*pRF/rhoRF)**0.5
        UnLF, UnRF = ULF.dotN(), URF.dotN()
        cF = (UnLF + cLF, UnRF + cLF, UnLF - cLF, UnRF - cLF)
        aF = cF[0].abs()
        for c in cF[1:]: aF = Field.max(aF, c.abs())
        aF.name = 'aF'

        # CFL based time step: sparse update?
        aF2 = Field.max((UnLF + aF).abs(), (UnRF - aF).abs())*0.5
        self.setDt(ad.value(aF2.field)/mesh.deltas)

        # flux reconstruction
        # phi (flux) for pressureInletVelocity
        rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF)
        self.flux = 2*rhoFlux/(rhoLF + rhoRF)
        rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF) - 0.5*aF*(rhoURF-rhoULF)
        rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF)
        pF = 0.5*(pLF + pRF)
        
        # viscous part
        TF = 0.5*(TLF + TRF)
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)
        UnF = 0.5*(UnLF + UnRF)
        UF = 0.5*(ULF + URF)
        #gradUTF = interpolate(grad(UF, ghost=True).transpose())
        gradUTF = interpolate(grad(UF, ghost=True))
        sigmaF = mu*(snGrad(U) + gradUTF.dotN() - (2./3)*gradUTF.trace()*mesh.Normals)

        # source terms
        source = self.source(self)
        #import pdb; pdb.set_trace()
        
        return [ddt(rho, self.dt) + div(rhoFlux) - source[0],
                ddt(rhoU, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF) - source[1],
                ddt(rhoE, self.dt) + div(rhoEFlux) - (laplacian(T, kappa) + div(sigmaF.dot(UF))) - source[2]]

    def boundary(self, rhoI, rhoUI, rhoEI):
        logger.info('correcting boundary')
        rhoN = Field(self.names[0], rhoI)
        rhoUN = Field(self.names[1], rhoUI)
        rhoEN = Field(self.names[2], rhoEI)
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        self.U.setInternalField(UN.field)
        self.T.setInternalField(TN.field)
        self.p.setInternalField(pN.field)
        return self.conservative(self.U, self.T, self.p)
    
if __name__ == "__main__":
    if len(sys.argv) > 2:
        case = sys.argv[1]
        time = float(sys.argv[2])
    else:
        pprint('WTF')
        exit()

    #solver = RCF(case, CFL=0.2, timeIntegrator='euler')
    solver = RCF(case, CFL=0.2, Cp=2.5, mu=lambda T: 0, timeIntegrator='euler')
    solver.run(startTime=time, nSteps=60000, writeInterval=1000)
