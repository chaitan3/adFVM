#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import numpad as ad
import time

from mesh import Mesh
from field import Field, CellField
from ops import  div, ddt, snGrad, laplacian, grad, implicit, explicit, forget, derivative
from ops import interpolate, upwind, TVD_dual
import utils


class Solver(object):
    def __init__(self, case, config):
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

    def primitive(self, rho, rhoU, rhoE):
        U = rhoU/rho
        E = rhoE/rho
        e = E - 0.5*U.magSqr()
        p = (self.gamma-1)*rho*e
        T = e*(1./self.Cv)
        return U, T, p

    def conservative(self, U, T, p):
        e = self.Cv*T
        rho = p/(e*(self.gamma-1))
        E = e + 0.5*U.magSqr()
        rhoU = rho*U
        rhoE = rho*E
        rho.name, rhoU.name, rhoE.name = 'rho', 'rhoU', 'rhoE'
        return rho, rhoU, rhoE
    
    def run(self, timeStep, nSteps, writeInterval=utils.LARGE, adjoint=False, objective=None):
        t, dt = timeStep
        mesh = self.mesh
        #initialize
        self.p = CellField.read('p', mesh, t)
        self.T = CellField.read('T', mesh, t)
        self.U = CellField.read('U', mesh, t)
        self.rho, self.rhoU, self.rhoE = self.conservative(self.U, self.T, self.p)
        self.fields = [self.rho, self.rhoU, self.rhoE]
        self.dt = dt
        self.adjoint = adjoint
        self.objective = objective
        print()
        mesh = self.mesh

        timeSteps = np.zeros((nSteps, 2))
        jacobians = np.zeros(nSteps).tolist()
        sensitivities = np.zeros(nSteps).tolist()
        for timeIndex in range(1, nSteps+1):
            timeSteps[timeIndex-1] = np.array([t, dt])
            t += self.dt
            t = round(t, 9)
            print('Simulation Time:', t, 'Time step:', self.dt)
            jacobians[timeIndex-1], sensitivities[timeIndex-1] = explicit(self.equation, self.boundary, [self.rho, self.rhoU, self.rhoE], self.dt)
            forget([self.p, self.T, self.U])
            if timeIndex % writeInterval == 0:
                self.rho.write(t)
                self.rhoU.write(t)
                self.rhoE.write(t)
                self.U.write(t)
                self.T.write(t)
                self.p.write(t)
            print()
        return timeSteps, jacobians, sensitivities

           
    def timeStep(self, aFbyD):
        self.dt = min(self.dt*self.stepFactor, self.CFL/np.max(aFbyD))

    def equation(self, rho, rhoU, rhoE):
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
        # zeroGrad interp on boundary for div and grad, ok?
        sigmaF = self.mu*(snGrad(U) + interpolate(grad(UF, ghost=True).transpose()).dotN() - (2./3)*interpolate(div(UnF, ghost=True))*mesh.Normals)
        
        return [ddt(rho, rho.old, self.dt) + div(rhoFlux),
                ddt(rhoU, rhoU.old, self.dt) + div(rhoUFlux) + grad(pF) - div(sigmaF),
                ddt(rhoE, rhoE.old, self.dt) + div(rhoEFlux) - (laplacian(e, self.alpha) + div(sigmaF.dot(UF)))]

    def boundary(self, rhoI, rhoUI, rhoEI):
        mesh = self.mesh
        rhoN = Field(self.rho.name, mesh, rhoI)
        rhoUN = Field(self.rhoU.name, mesh, rhoUI)
        rhoEN = Field(self.rhoE.name, mesh, rhoEI)
        UN, TN, pN = self.primitive(rhoN, rhoUN, rhoEN)
        self.U.setInternalField(UN.field)
        self.T.setInternalField(TN.field)
        self.p.setInternalField(pN.field)
        rhoN, rhoUN, rhoEN = self.conservative(self.U, self.T, self.p)
        if self.adjoint:
            jacobian = derivative((rhoN, rhoUN, rhoEN), (self.rho, self.rhoU, self.rhoE))
        else:
            jacobian = 0
        if self.objective is not None:
            sensitivity = derivative(self.objective(self), (self.rho, self.rhoU, self.rhoE)).todense()
        else:
            sensitivity = 0
        self.rho.field, self.rhoU.field, self.rhoE.field = rhoN.field, rhoUN.field, rhoEN.field
        return jacobian, sensitivity
    

if __name__ == "__main__":

    #solver = Solver('tests/cylinder/', {'R': 8.314, 'Cp': 1006., 'gamma': 1.4, 'mu': 2.5e-5, 'Pr': 0.7, 'CFL': 0.2})
    #solver.run([1.8, 1e-9], 10000, 100)
    solver = Solver('tests/forwardStep/', {'R': 8.314, 'Cp': 2.5, 'gamma': 1.4, 'mu': 0, 'Pr': 0.7, 'CFL': 0.2})
    solver.run([0, 1e-4], 40000, 500)

