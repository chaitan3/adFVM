#!/usr/bin/python2
from __future__ import print_function
import numpy as np
import numpad as ad
import time

from mesh import Mesh
from field import Field, FaceField
from ops import  div, ddt, solve, laplacian, grad
from ops import interpolate, upwind

gamma = 1.4
#mu = 2.5e-5
mu = 0.
Pr = 0.7
alpha = mu/Pr
Cp = 1007
Cv = Cp/gamma


case = 'shockTube/'
mesh = Mesh(case)

t = 0
dt = 1e-4
writeInterval = 5

#initialize

p = Field.read('p', mesh, t)
T = Field.read('T', mesh, t)
U = Field.read('U', mesh, t)
e = Cv*T

def primitive(rho, rhoU, rhoE):
    U = rhoU/rho
    E = rhoE/rho
    e = E - 0.5*U.mag()
    p = (gamma-1)*rho*e
    return U, e, p

def conservative(U, e, p):
    rho = p/(e*(gamma-1))
    E = e + 0.5*U.mag()
    return rho, rho*U, rho*E

rho, rhoU, rhoE = conservative(U, e, p)
rho.name = 'rho'
rhoU.name = 'rhoU'
rhoE.name = 'rhoE'

for timeIndex in range(1, 300):

    print('Simulation Time:', t, 'Time step:', dt)

    rho0 = Field.copy(rho)
    rhoU0 = Field.copy(rhoU)
    rhoE0 = Field.copy(rhoE)

    def eq(rho, rhoU, rhoE):
        
        #best way?
        Uf = interpolate(rhoU/rho)

        rhof = upwind(rho, Uf)
        rhoUf = upwind(rhoU, Uf)
        rhoEf = upwind(rhoE, Uf)

        # fancy interpolatex computation?
        Uf, ef, pf = primitive(rhof, rhoUf, rhoEf)
        U, e, p = primitive(rho, rhoU, rhoE)

        return [ddt(rho, rho0, dt) + div(rhof, Uf),
                ddt(rhoU, rhoU0, dt) + div(rhoUf, Uf) + grad(pf) - (laplacian(U, mu)), #+ div(grad(U))
                ddt(rhoE, rhoE0, dt) + div(rhoEf + pf, Uf) - (laplacian(e, alpha) )] #+ div(sigma, Uf))]
    
    solve(eq, [rho, rhoU, rhoE])

    t += dt
    t = round(t, 6)
    print()

    if timeIndex % writeInterval == 0:
        rho.write(t)
        rhoU.write(t)
        rhoE.write(t)
   
