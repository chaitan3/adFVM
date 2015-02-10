#!/usr/bin/python2
import numpy as np
import sys
import time

from field import Field, CellField, IOField
from op import  div, snGrad, grad, ddt, laplacian
from solver import Solver
from interp import central, TVD_dual

from config import ad, Logger, T
import config
from parallel import pprint
logger = Logger(__name__)
import config, parallel

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
        #WTF is this needed?
        rho.field = rho.field.reshape((-1,1))
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

    def initFields(self, t):
        self.p = IOField.read('p', self.mesh, t)
        self.T = IOField.read('T', self.mesh, t)
        self.U = IOField.read('U', self.mesh, t)
        self.p.complete()
        self.T.complete()
        self.U.complete()
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
        from config import T as theano

        # in interpolation phi is full with additional second layer, gradField is full
        gradRho = grad(central(rhoP, paddedMesh), ghost=True)
        gradRhoU = grad(central(rhoUP, paddedMesh), ghost=True)
        gradRhoE = grad(central(rhoEP, paddedMesh), ghost=True)

        # phi is in paddedMesh form, needs to be copied to regular
        # phi from phiPaddedMesh
        UP, TP, pP = self.primitive(rhoP, rhoUP, rhoEP)
        rho = CellField.getOrigField(rhoP)
        rhoU = CellField.getOrigField(rhoUP)
        rhoE = CellField.getOrigField(rhoEP)
        U = CellField.getOrigField(UP)
        T = CellField.getOrigField(TP)
        #theano.printing.Print('sdsd')(pP.field.shape)

        # interpolation
        rhoLF, rhoRF = TVD_dual(rho, gradRho)
        rhoULF, rhoURF = TVD_dual(rhoU, gradRhoU)
        rhoELF, rhoERF = TVD_dual(rhoE, gradRhoE)
        ULF, TLF, pLF = self.primitive(rhoLF, rhoULF, rhoELF)
        URF, TRF, pRF = self.primitive(rhoRF, rhoURF, rhoERF)

        # numerical viscosity
        # no TVD_dual for c in parallel
        cP = (self.gamma*pP/rhoP)**0.5
        c = CellField.getOrigField(cP)
        gradC = grad(central(cP, paddedMesh), ghost=True)
        cLF, cRF = TVD_dual(c, gradC)
        #cLF, cRF = (self.gamma*pLF/rhoLF)**0.5, (self.gamma*pRF/rhoRF)**0.5
        UnLF, UnRF = ULF.dotN(), URF.dotN()
        #cF = (UnLF + cLF, UnRF + cLF, UnLF - cLF, UnRF - cLF)
        #aF = cF[0].abs()
        #for c in cF[1:]: aF = Field.max(aF, c.abs())
        Z = Field('Z', ad.zeros((mesh.nFaces, 1)), (1,))
        apF = Field.max(Field.max(UnLF + cLF, UnRF + cRF), Z)
        amF = Field.min(Field.min(UnLF - cLF, UnRF - cRF), Z)
        aF = Field.max(apF.abs(), amF.abs())
        aF.name = 'aF'

        # CFL based time step: sparse update?
        aF2 = Field.max((UnLF + aF).abs(), (UnRF - aF).abs())*0.5
        self.dtc = self.CFL/ad.max(aF2.field/mesh.deltas)

        # flux reconstruction
        # phi (flux) for pressureInletVelocity
        rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF)
        self.flux = 2.*rhoFlux/(rhoLF + rhoRF)
        rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF) - 0.5*aF*(rhoURF-rhoULF)
        rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF)
        pF = 0.5*(pLF + pRF)
        
        UF = 0.5*(ULF + URF)
        #gradUTF = interpolate(grad(UF, ghost=True))
        rhoR = Field(rho.name, rho.field.reshape((-1, 1, 1)), (1, 1))
        gradUT = (rhoR*gradRhoU.transpose()-gradRho.outer(rhoU))/(rhoR*rhoR)
        gradUTF = central(gradUT, rho.mesh)

        # viscous part
        TF = 0.5*(TLF + TRF)
        mu = self.mu(TF)
        kappa = self.kappa(mu, TF)
        UnF = 0.5*(UnLF + UnRF)
        sigmaF = (snGrad(U) + gradUTF.dotN() - (2./3)*mesh.Normals*gradUTF.trace())*mu

        # source terms
        source = self.source(self)
        #import pdb; pdb.set_trace()
        
        return [ddt(rho, self.dt) + div(rhoFlux) - source[0],
                ddt(rhoU, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF) - source[1],
                ddt(rhoE, self.dt) + div(rhoEFlux) - (laplacian(T, kappa) + div(sigmaF.dot(UF))) - source[2]]

    def boundary(self, rhoI, rhoUI, rhoEI):
        logger.info('correcting boundary')
        rhoN = Field(self.names[0], rhoI, self.dimensions[0])
        rhoUN = Field(self.names[1], rhoUI, self.dimensions[1])
        rhoEN = Field(self.names[2], rhoEI, self.dimensions[2])
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        U = CellField('U', UN.field, self.U.dimensions, self.U.boundary, ghost=True)
        T = CellField('T', TN.field, self.T.dimensions, self.T.boundary, ghost=True)
        p = CellField('p', pN.field, self.p.dimensions, self.p.boundary, ghost=True)
        return self.conservative(U, T, p)
    
if __name__ == "__main__":
    if len(sys.argv) > 2:
        case = sys.argv[1]
        time = float(sys.argv[2])
    else:
        pprint('WTF')
        exit()

    #solver = RCF(case, CFL=0.6)
    solver = RCF(case, CFL=0.2, Cp=2.5, mu=lambda T: 1e-30*T, timeIntegrator='euler')
    solver.run(startTime=time, dt=1e-4, nSteps=60000, writeInterval=100)
