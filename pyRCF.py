#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import sys
import time

from mesh import Mesh
from field import Field, CellField
from ops import  div, ddt, snGrad, laplacian, grad, implicit, explicit, forget, strip
from ops import interpolate, TVD_dual

from utils import ad, pprint
from utils import Logger
logger = Logger(__name__)
import utils


class Solver(object):
    def __init__(self, case, config):
        logger.info('initializing solver for {0}'.format(case))
        self.R = config['R']
        self.Cp = config['Cp']
        self.gamma = config['gamma']
        self.Cv = self.Cp/self.gamma
        self.mu = config['mu']
        self.Pr = config['Pr']
        self.alpha = self.mu/self.Pr

        self.mesh = Mesh(case)

        self.stepFactor = 1.2
        self.CFL = config['CFL']

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
    
    def run(self, timeStep, nSteps, writeInterval=utils.LARGE, adjoint=False, objective=lambda x: 0, perturb=None):
        logger.info('running solver for {0}'.format(nSteps))
        t, dt = timeStep
        mesh = self.mesh
        #initialize
        self.p = CellField.read('p', mesh, t)
        self.T = CellField.read('T', mesh, t)
        self.U = CellField.read('U', mesh, t)
        fields = self.conservative(self.U, self.T, self.p)
        if perturb is not None:
            perturb(fields)
        self.dt = dt
        pprint()
        mesh = self.mesh

        timeSteps = np.zeros((nSteps + 1, 2))
        result = 0
        solutions = []
        for timeIndex in range(1, nSteps+1):
            if adjoint:
                oldFields = strip(fields)
                solutions.append([fields, oldFields])
                fields = explicit(self.equation, self.boundary, oldFields, self)
            else:
                result += objective(fields)
                fields = explicit(self.equation, self.boundary, fields, self)
                timeSteps[timeIndex-1] = np.array([t, self.dt])
                forget([self.p, self.T, self.U])
            t += self.dt
            t = round(t, 9)
            pprint('Simulation Time:', t, 'Time step:', self.dt)
            if timeIndex % writeInterval == 0:
                for phi in fields:
                    phi.write(t)
                self.U.write(t)
                self.T.write(t)
                self.p.write(t)
            pprint()
        if adjoint:
            solutions.append([fields, strip(fields)])
            return solutions
        else:
            timeSteps[nSteps] = np.array([t, self.dt])
            result += objective(fields)
            return timeSteps, result
           
    def timeStep(self, aFbyD):
        logger.info('computing new time step')
        self.dt = min(self.dt*self.stepFactor, self.CFL/utils.max(aFbyD))

    def equation(self, rho, rhoU, rhoE):
        logger.info('computing RHS/LHS')
        mesh = self.mesh

        rhoLF, rhoRF = TVD_dual(rho)
        rhoULF, rhoURF = TVD_dual(rhoU)
        rhoELF, rhoERF = TVD_dual(rhoE)

        ULF, TLF, pLF = self.primitive(rhoLF, rhoULF, rhoELF)
        URF, TRF, pRF = self.primitive(rhoRF, rhoURF, rhoERF)
        U, T, p = self.primitive(rho, rhoU, rhoE)
        e = self.Cv*T

        # numerical viscosity
        cLF, cRF = (self.gamma*pLF/rhoLF)**0.5, (self.gamma*pRF/rhoRF)**0.5
        UnLF, UnRF = ULF.dotN(), URF.dotN()
        cF = (UnLF + cLF, UnRF + cLF, UnLF - cLF, UnRF - cLF)
        aF = cF[0].abs()
        for c in cF[1:]: aF = Field.max(aF, c.abs())
        aF.name = 'aF'

        # CFL based time step: sparse update?
        aF2 = Field.max((UnLF + aF).abs(), (UnRF - aF).abs())*0.5
        self.timeStep(ad.value(aF2.field)/mesh.deltas)

        # flux reconstruction
        rhoFlux = 0.5*(rhoLF*UnLF + rhoRF*UnRF) - 0.5*aF*(rhoRF-rhoLF)
        rhoUFlux = 0.5*(rhoULF*UnLF + rhoURF*UnRF) - 0.5*aF*(rhoURF-rhoULF)
        rhoEFlux = 0.5*((rhoELF + pLF)*UnLF + (rhoERF + pRF)*UnRF) - 0.5*aF*(rhoERF-rhoELF)
        pF = 0.5*(pLF + pRF)
        
        # viscous part
        UnF = 0.5*(UnLF + UnRF)
        UF = 0.5*(ULF + URF)
        sigmaF = self.mu*(snGrad(U) + interpolate(grad(UF, ghost=True).transpose()).dotN() - (2./3)*interpolate(div(UnF, ghost=True))*mesh.Normals)
        
        #return [ddt(rho, rho.old, self.dt) + div(rhoFlux),
        #        ddt(rhoU, rhoU.old, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF),
        #        ddt(rhoE, rhoE.old, self.dt) + div(rhoEFlux) - (laplacian(e, self.alpha) + div(sigmaF.dot(UF)))]
        return [div(rhoFlux),
                div(rhoUFlux) + grad(pF) - div(sigmaF),
                div(rhoEFlux) - (laplacian(e, self.alpha) + div(sigmaF.dot(UF)))]



    def boundary(self, rhoI, rhoUI, rhoEI):
        logger.info('correcting boundary')
        mesh = self.mesh
        rhoN = Field(self.names[0], mesh, rhoI)
        rhoUN = Field(self.names[1], mesh, rhoUI)
        rhoEN = Field(self.names[2], mesh, rhoEI)
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        self.U.setInternalField(UN.field)
        self.T.setInternalField(TN.field)
        self.p.setInternalField(pN.field)
        return self.conservative(self.U, self.T, self.p)
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        case = sys.argv[1]
    else:
        case = 'tests/cylinder'

    solver = Solver(case, {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 2.5e-5, 'Pr': 0.7, 'CFL': 0.2})
    solver.run([0, 1e-3], 1000, 100)
    #solver = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0, 'Pr': 0.7, 'CFL': 0.2})
    #solver.run([0, 1e-3], 40000, 500)

