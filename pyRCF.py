#!/usr/bin/python2
import numpy as np
import sys
import time

import config
from config import ad
from parallel import pprint

from field import Field, CellField, IOField
from op import  div, snGrad, grad, ddt, laplacian, internal_sum
from solver import Solver
from interp import central, TVD_dual, TVD2_dual
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
                             'timeIntegrator': 'SSPRK',
                             'riemannSolver': 'kurganovTadmor',
                             'source': lambda x: [0, 0, 0]
                        })

    def __init__(self, case, **userConfig):
        super(RCF, self).__init__(case, **userConfig)

        self.Cv = self.Cp/self.gamma
        self.kappa = lambda mu, T: mu*(self.Cp/self.Pr)
        self.names = ['rho', 'rhoU', 'rhoE']
        self.dimensions = [(1,), (3,), (1,)]
        self.riemann = getattr(riemann, self.riemannSolver)

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
        rhoU = U*rho
        rhoE = rho*E
        rho.name, rhoU.name, rhoE.name = self.names
        return rho, rhoU, rhoE

    def getRho(self, T, p):
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        return rho

    def getBCFields(self):
        return self.p, self.T, self.U

    def initFields(self, t):
        self.p = IOField.read('p', self.mesh, t)
        self.T = IOField.read('T', self.mesh, t)
        self.U = IOField.read('U', self.mesh, t)
        if not hasattr(self, "pfunc"):
            self.pfunc = self.Tfunc = self.Ufunc = None
        self.pfunc = self.p.complete(self.pfunc)
        self.Tfunc = self.T.complete(self.Tfunc)
        self.Ufunc = self.U.complete(self.Ufunc)
        return self.conservative(self.U, self.T, self.p)
    
    def writeFields(self, fields, t):
        for phi in fields:
            phi.write(t)
        U, T, p = self.primitive(*fields)
        self.U.field = U.field
        self.T.field = T.field
        self.p.field = p.field
        self.U.write(t)
        self.T.write(t)
        self.p.write(t)
           
    def equation(self, rhoP, rhoUP, rhoEP, exit=False):
        logger.info('computing RHS/LHS')
        mesh = self.mesh
        paddedMesh = mesh.paddedMesh
        gamma = self.gamma

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
        #rhoLF, rhoRF = TVD2_dual(rho, gradRho)
        #rhoULF, rhoURF = TVD2_dual(rhoU, gradRhoU)
        #rhoELF, rhoERF = TVD2_dual(rhoE, gradRhoE)
        #ULF, TLF, pLF = self.primitive(rhoLF, rhoULF, rhoELF)
        #URF, TRF, pRF = self.primitive(rhoRF, rhoURF, rhoERF)

        # gradient evaluated using gauss integration rule
        gradp = grad(central(pP, paddedMesh), ghost=True)
        gradT = grad(central(TP, paddedMesh), ghost=True)
        gradU = grad(central(UP, paddedMesh), ghost=True)

        # face reconstruction
        pLF, pRF = TVD2_dual(p, gradp)
        TLF, TRF = TVD2_dual(T, gradT)
        ULF, URF = TVD2_dual(U, gradU)
        rhoLF, rhoULF, rhoELF = self.conservative(ULF, TLF, pLF)
        rhoRF, rhoURF, rhoERF = self.conservative(URF, TRF, pRF)

        # scalar Dissipaton 
        # no TVD_dual for c in parallel
        #cP = (gamma*pP/rhoP)**0.5
        #c = CellField.getOrigField(cP)
        #gradC = grad(central(cP, paddedMesh), ghost=True)
        #cLF, cRF = TVD_dual(c, gradC)

        #UnLF, UnRF = ULF.dotN(), URF.dotN()
        #Z = Field('Z', ad.bcalloc(config.precision(0.), (mesh.nFaces, 1)), (1,))
        #apF = Field.max(Field.max(UnLF + cLF, UnRF + cRF), Z)
        #amF = Field.min(Field.min(UnLF - cLF, UnRF - cRF), Z)
        #aF = Field.max(apF.abs(), amF.abs())

        #rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF)
        #rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF + (pLF + pRF)*mesh.Normals) - 0.5*aF*(rhoURF-rhoULF)
        #rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF)
 
        #rhoFlux, rhoUFlux, rhoEFlux = self.riemann(rhoLF, rhoRF, rhoULF, rhoURF, rhoELF, rhoERF, UnLF, UnRF, aF)

        # roe Flux
        rhoUnLF, rhoUnRF = rhoLF*ULF.dotN(), rhoRF*URF.dotN()
        hLF = gamma*pLF/((gamma-1)*rhoLF) + 0.5*ULF.magSqr()
        hRF = gamma*pRF/((gamma-1)*rhoRF) + 0.5*URF.magSqr()

        rhoFlux = 0.5*(rhoUnLF + rhoUnRF)
        rhoUFlux = 0.5*(rhoUnLF*ULF + rhoUnRF*URF + (pLF + pRF)*mesh.Normals)
        rhoEFlux = 0.5*(rhoUnLF*hLF + rhoUnRF*hRF)

        sqrtRhoLF, sqrtRhoRF = rhoLF**0.5, rhoRF*0.5
        divRhoF = sqrtRhoLF + sqrtRhoRF
        UF = (ULF*sqrtRhoLF + URF*sqrtRhoRF)/divRhoF
        hF = (hLF*sqrtRhoLF + hRF*sqrtRhoRF)/divRhoF

        qF = 0.5*UF.magSqr()
        a2F = (gamma-1)*(hF-qF)
        aF = a2F**0.5
        UnF = UF.dotN()

        drhoF = rhoRF - rhoLF 
        drhoUF = rhoRF*URF - rhoLF*ULF
        drhoEF = (hRF*rhoRF-pRF)-(hLF*rhoLF-pLF)

        lam1, lam2, lam3 = UnF.abs(), (UnF + aF).abs(), (UnF - aF).abs()

        eps = 0.5*(rhoUnLF/rhoLF - rhoUnRF/rhoRF).abs()
        eps += 0.5*((gamma*pLF/rhoLF)**0.5 - (gamma*pRF/rhoRF)**0.5).abs()

        lam1 = Field.switch(ad.lt(lam1.field, 2.*eps.field), 0.25*lam1*lam1/eps + eps, lam1)
        lam2 = Field.switch(ad.lt(lam2.field, 2.*eps.field), 0.25*lam2*lam2/eps + eps, lam2)
        lam3 = Field.switch(ad.lt(lam3.field, 2.*eps.field), 0.25*lam3*lam3/eps + eps, lam3)

        abv1 = 0.5*(lam2 + lam3)
        abv2 = 0.5*(lam2 - lam3)
        abv3 = abv1 - lam1
        abv4 = (gamma-1)*(qF*drhoF - UF.dot(drhoUF) + drhoEF)
        abv5 = UnF*drhoF - drhoUF.dotN()
        abv6 = abv3*abv4/a2F - abv2*abv5/aF
        abv7 = abv3*abv5 - abv2*abv4/aF

        rhoFlux -= 0.5*(lam1*drhoF + abv6)
        rhoUFlux -= 0.5*(lam1*drhoUF + UF*abv6 - abv7*mesh.Normals)
        rhoEFlux -= 0.5*(lam1*drhoEF + hF*abv6 - UnF*abv7)
        
        #rhoR = Field(rho.name, rho.field.reshape((mesh.nCells, 1, 1)), (1, 1))
        #gradUT = (rhoR*gradRhoU.transpose()-gradRho.outer(rhoU))/(rhoR*rhoR)
        # TODO: change reconstruction of gradient on face
        gradUT = gradU.transpose()
        gradUTF = central(gradUT, mesh)

        # viscous part
        #pF = 0.5*(pLF + pRF)
        UF = 0.5*(ULF + URF)
        TF = 0.5*(TLF + TRF)
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)
        sigmaF = (snGrad(U) + gradUTF.dotN() - (2./3)*mesh.Normals*gradUTF.trace())*mu
        # TODO: check laplacian and viscous terms

        # source terms
        source = self.source(self)

        # CFL based time step
        aF = UnF.abs() + aF
        self.dtc = 2*self.CFL/internal_sum(aF, mesh, absolute=True)
        
        return [div(rhoFlux) - source[0],
                #ddt(rhoU, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF) - source[1],
                div(rhoUFlux) - div(sigmaF) - source[1],
                div(rhoEFlux) - (laplacian(T, kappa) + div(sigmaF.dot(UF))) - source[2]]

    def boundary(self, rhoI, rhoUI, rhoEI):
        logger.info('correcting boundary')
        rhoN = Field(self.names[0], rhoI, self.dimensions[0])
        rhoUN = Field(self.names[1], rhoUI, self.dimensions[1])
        rhoEN = Field(self.names[2], rhoEI, self.dimensions[2])
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        self.U.phi.setInternalField(UN.field, reset=True)
        self.T.phi.setInternalField(TN.field, reset=True)
        self.p.phi.setInternalField(pN.field, reset=True)
        return self.conservative(self.U.phi, self.T.phi, self.p.phi)
    
if __name__ == "__main__":
    if len(sys.argv) > 2:
        case = sys.argv[1]
        time = float(sys.argv[2])
    else:
        pprint('WTF')
        exit()

    #solver = RCF(case, CFL=0.5)
    #solver.run(startTime=time, dt=1e-9, nSteps=60000, writeInterval=200)
    solver = RCF(case, CFL=0.7, Cp=2.5, mu=lambda T: config.VSMALL*T)
    solver.run(startTime=time, dt=1e-4, nSteps=60000, writeInterval=200)
